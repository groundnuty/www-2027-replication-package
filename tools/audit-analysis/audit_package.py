"""The post-run audit's annotation package (S3.1; pre-registration §5 "Audits", H4 and H6), built from a drawn sample
(``harness/analysis/audit_sample.py``) and the outputs it names, by ``paper/audit-protocol.md`` at b9777bc
(anonymous/science-repo): "What annotators see (blinding)" and the codebook.

    uv run python -m harness.analysis.audit_package --sample SAMPLE.json --records OUT [OUT ...] \\
        --matrix MATRIX.jsonl.gz [...] --questions QUESTIONS.jsonl [...] --out PACKAGE_DIR --key-out KEY.json \\
        [--annotators A B] [--practice] [--private]
    uv run python -m harness.analysis.audit_package --check-sheet FILLED.csv --package PACKAGE_DIR --annotator NAME \
        [--complete]

**What an item shows** (``SHOWN``, an allowlist: nothing else of a record, a matrix row or the sample is rendered):
the question, as the agent was given it (the record's transcript message of kind "question"); the gold query and the
gold answer (the question set's ``gold_sparql`` and ``gold_rows``); the agent's final query and answer. The final
query is rule 22's checked query (harness/checkloop/checked.py), the query the registered offline checker read: the
matrix rows' ``checked_query`` (variant primary; the design log), recomputed here from the record by the frozen
``checked_output`` and refused unless the two agree. When no executed query matched the answer (``fallback``) or none
ran (``none``), a note says so. The answer is the initial output's ``final_rows``. **Kind B** adds the flag message:
the curated row's ``fired`` lines through the frozen ``render_lines``, the source-neutral text the agent loop feeds
back (rule 38). **Kind C** adds the ontology text rendered into the agent's prompt (the block named by the item's
``ontology_block_sha256``, with its framing) and asks for the witness. Never shown: the model, its group, the cohort
or arm, the graph's name, run, question and base question ids, the cell, any status, fired check, reason or
per-source verdict, and the record's and prompt's hashes.

**Scans before anything is written.** Every annotator file is searched for each identifier-like value (six
characters or more with a digit or one of ``-_./:``) of every item's sample entry (the graph's name aside: the protocol
does not blind it, and a shown ontology text may name it), record hash and matrix rows (check ids, reasons, sources,
arms); a plain word such as a cohort's "curated" can be shown content (the curated ontology text), so it is kept out
by the allowlist alone, which the tests prove with sentinels. The repository's ontology blocks hold none of the
registered identifier-like values (a test checks the corpus). The model-written parts, the agent's final query and
answer, are searched for the families' names (``FAMILY_WORDS``): a model naming itself would unblind the item, so the
build refuses until the protocol's owner rules and ``--allow-family-word RUN_ID:WORD`` lets that word through for
that output (an organism named "llama" in a question the query quotes); the manifest lists each word let through.

**Checks** (each refusal names the run id and the field, never content; a private build's masks the ids): one record
per item, agreeing with the sample on graph, model, question, base question and split, and for C on the prompt link
(``record_sha256``, condition, block), and C's cohort the curated prompt's; a submitted initial output whose score's
``exact`` is the kind's (B correct, A and C wrong), as is every matrix row's ``correct`` (a boolean, never missing,
as check_record writes it for a submitted output); primary matrix
rows (rows of several matrices must agree, and an output's arms must name one checked query) with a known status
(``LETTER``): B's curated row flagged, C's a pass, A's not flagged and no residue source's row flagged (A admits an
unknown, as the draw does: U is not a flag); the question resolved among ``--questions`` by its id, the record's
question text and its initial score's ``n_gold_rows``, with every surviving candidate agreeing on the gold query and
rows.

**Practice-only items** (a practice sample only; refused in any other): a sample item marked ``practice_only`` (the
practice draw's top-up of kind C, outside the registered population) is checked as its kind except for C's curated
cohort and curated pass, and the manifest lists those items apart from the strict ones.

**Forms and the codebook** (``CODEBOOK``): the codebook's fields and labels verbatim, under neutral form names
("Form A", "B", "C"). The guide names neither the kinds ("residue", "false flag", "shared premise") nor the
populations, since telling an annotator that no checker flagged an item could bias "structurally detectable" (H4's
measure). Kind C's columns are the sampler's ``C_RESPONSE`` names. An annotator who knows the protocol can still tell
the populations from the forms: that is the protocol's design (one codebook per kind), not something a rendering hides.

**Order** (the protocol shuffles across kinds and graphs, and says nothing on the annotators' orders, so each
annotator gets an independent order): a fresh 128-bit shuffle seed per build (``secrets``; ``--shuffle-seed`` takes
only a key's 32 hex digits, to rebuild), kept only in the key. Items are ranked by sha256("<seed>|id|<run id>") and
numbered in that order (``P001`` in a practice package, ``T001`` otherwise); each annotator's order ranks the ids by
sha256("<seed>|order|<annotator>|<item id>"). The manifest records the seed's sha256 only. **The key is sealed
against the draw's seed, not against the sample:** the sample names each item's question and model, and a question's
text identifies it, so the annotators must not have the sample (or the key) until their sheets are locked; the
manifest's ``sample_sha256`` commits to it meanwhile.

**Outputs.** ``--out``: ``manifest.json`` (inputs' hashes, counts, the seed's and the key's sha256, every file's
sha256), ``HANDLING.md`` (for the lead: what to hand out, what to withhold, that annotators open neither the
experiments repository, its records nor the cluster until their coding ends, and that the real package is built and
delivered off GitHub from the cluster) and ``annotators/<name>/`` with ``README.md`` (the guide and codebook),
``items.md`` and ``items.jsonl`` (the renderings in that annotator's order) and ``sheet.csv`` (one row per item, the
codebook's columns; "-" where a column is not the form's). ``--key-out`` (mode 600, outside ``--out``): the sealed key,
mapping each item id back to the sample's item and its record's sha256, the seed and the orders. Both are staged beside
their targets and moved into place only when everything is written. ``--private`` (a sample whose header says private):
both paths outside this repository, under directories open to their owner only, an existing ``--out`` too; directories
700, files 600; nothing of the items, ids or counts, printed. A public build refuses a private sample and any item of a
private graph.

Exit 0: written; 2: refused, and nothing is written. ``--check-sheet`` (against the package's own sheet, checked
against its manifest): 0 when the filled sheet has no problem, else 1.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import re
import secrets
import shutil
import stat
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.agent import prompts
from harness.analysis import audit_sample as au
from harness.checkloop.checked import checked_output, rules_of
from harness.checkloop.feedback import render_lines
from harness.checkloop.ledger import LETTER
from harness.checkloop.matrix import initial_of, load_raw_records
from harness.terms import IRI, BNode, Literal, TermError, row_from_json

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_SCHEMA = "h4-audit-package/1"
KEY_SCHEMA = "h4-audit-key/1"
PROTOCOL = au.PROTOCOL
ANNOTATORS = ("annotator-1", "annotator-2")
CURATED_ARM = au.CURATED
PRIMARY = "primary"  # the matrix's frozen online setting (harness/checkloop/matrix.py VARIANTS)
FLAGGED, PASSES = "flagged", ("substantive_pass", "vacuous_pass")
SHOWN = ("question", "gold_query", "gold_answer", "agent_query", "agent_answer")
SHOWN_BY_KIND = {"A": SHOWN, "B": (*SHOWN, "flag_message"), "C": (*SHOWN, "ontology_text")}
NOT_SHOWN = ("model", "model_group", "cohort", "arm", "graph", "run_id", "question_id", "base_question_id", "cell",
             "status", "fired_checks", "reason", "per_source", "record_sha256", "prompt_id", "prompt_sha256")
QUERY_NOTES = {
    "aligned": None,
    "fallback": "No query the agent ran matched its submitted answer under the registered selection rule; this is "
                "the last query it ran before answering.",
    "none": "The agent ran no query before answering.",
}
# the panel's families and their models' names (data/analysis/prereg-config.json), and other models' that name
# themselves; words a query uses otherwise (meta, google, coder) are left out
FAMILY_WORDS = ("claude", "anthropic", "sonnet", "haiku", "opus", "qwen", "glm", "zhipu", "gemma", "gemini",
                "deepseek", "llama", "gpt", "openai", "mistral")
MAX_ROWS_SHOWN = 200  # in items.md; items.jsonl holds every row
NA = "-"
SEED_RE = re.compile(r"[0-9a-f]{32}")
# a record as RunRecord.dumps writes it: schema_version, then run_id
_CANONICAL_RUN_ID = re.compile(r'\s*\{\s*"schema_version":\s*"[^"\\]*",\s*"run_id":\s*"([^"\\]+)"')


@dataclass(frozen=True)
class Field:
    column: str
    question: str
    labels: tuple[str, ...] | None  # None: free text
    note: str | None = None  # the column of the note an "other" needs


YNU = ("yes", "no", "unsure")
# audit-protocol.md "Codebook (one primary code per field)", labels verbatim; C's columns are the sampler's C_RESPONSE
CODEBOOK: dict[str, tuple[Field, ...]] = {
    "A": (Field("error_type", "Error type",
                ("aggregation", "constant", "filter_scope", "path", "shape", "gold_or_question", "scorer", "other"),
                note="error_type_note"),
          Field("structurally_detectable", "Structurally detectable?", YNU)),
    "B": (Field("cause", "Cause",
                ("undeclared_term", "closed_world", "data_clash", "schema_too_narrow", "checker_defect", "other"),
                note="cause_note"),
          Field("answer_really_correct", "Was the agent's answer really correct?", YNU)),
    "C": (Field("follows_from_ontology",
                "Does the wrong answer follow from a statement in the ontology shown in the prompt?", YNU),
          Field("witness", "The quoted statement (the witness)", None),
          Field("statement_wrong_or_misleading", "If yes, is that statement wrong or misleading for this data?", YNU)),
}
assert [f.column for f in CODEBOOK["C"]] == list(au.C_RESPONSE)
FORM_COLUMNS = {k: [c for f in fs for c in (f.column, f.note) if c] for k, fs in CODEBOOK.items()}
SHEET_COLUMNS = ["order", "item_id", "form", *(c for k in au.KINDS for c in FORM_COLUMNS[k]), "comment"]
FORM_TITLES = {
    "A": "Form A: an answer scored wrong",
    "B": "Form B: an answer scored correct, with a checker's message",
    "C": "Form C: an answer scored wrong, with the ontology text from the agent's prompt",
}


class PackageError(ValueError):
    """An input or a request the builder refuses. Messages name run ids and fields, never content."""


# --- inputs -------------------------------------------------------------------------------------------------------

def _lines(path: Path) -> Iterable[str]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        yield from f


MATRIX_READ = ("checked_query", "checked_query_status", "status", "fired", "correct")


def matrix_rows(paths: Sequence[Path], run_ids: Iterable[str]) -> dict[str, dict[str, dict[str, Any]]]:
    """Each wanted output's primary matrix rows by arm (the registered offline checker's, the design log). Rows of one
    output and arm in several matrices must agree on what the package reads, and an output's arms on its checked
    query (computed once per record, harness/checkloop/matrix.py check_record)."""
    wanted = set(run_ids)
    found: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for path in paths:
        for line in _lines(path):
            if not line.strip():
                continue
            row = json.loads(line)
            rid, arm = row.get("run_id"), row.get("arm")
            if rid not in wanted or row.get("variant") != PRIMARY:
                continue
            if arm in found[rid] and any(found[rid][arm].get(k) != row.get(k) for k in MATRIX_READ):
                raise PackageError(f"output {rid}: two matrix rows of arm {arm} disagree")
            found[rid][arm] = row
    if missing := sorted(r for r in wanted if not found.get(r)):
        raise PackageError(f"output {missing[0]} (and {len(missing) - 1} more) has no primary matrix row")
    for rid, arms in found.items():
        if len({json.dumps([r.get("checked_query"), r.get("checked_query_status")]) for r in arms.values()}) > 1:
            raise PackageError(f"output {rid}: its matrix rows name two checked queries")
    return dict(found)


def records_of(raw_records: Iterable[str], run_ids: Iterable[str]) -> dict[str, tuple[dict[str, Any], str]]:
    """Each wanted output's run record and the sha256 of its stored bytes, read one at a time. A record laid out as
    RunRecord.dumps writes it names its run id first and is skipped unparsed when that id is not wanted; any other is
    parsed."""
    wanted, found = set(run_ids), {}
    for raw in raw_records:
        m = _CANONICAL_RUN_ID.match(raw)
        if m and m.group(1) not in wanted:
            continue
        rec = json.loads(raw)
        rid = rec.get("run_id")
        if rid not in wanted:
            continue
        if rid in found:
            raise PackageError(f"output {rid} has two run records")
        found[rid] = (rec, hashlib.sha256(raw.encode("utf-8")).hexdigest())
    if missing := sorted(wanted - set(found)):
        raise PackageError(f"output {missing[0]} (and {len(missing) - 1} more) has no run record among --records")
    return found


def question_sets(paths: Sequence[Path]) -> dict[str, list[dict[str, Any]]]:
    """Every question of the files given, by id (an id may appear in several files)."""
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        for line in _lines(path):
            if line.strip():
                q = json.loads(line)
                by_id[q["id"]].append(q)
    return by_id


def question_text(rec: Mapping[str, Any]) -> str:
    for m in rec.get("transcript") or ():
        if m.get("role") == "user" and m.get("kind") == "question":
            return m["text"]
    raise PackageError(f"output {rec.get('run_id')}: its transcript has no question message")


def resolve_question(rid: str, qid: str, text: str, n_gold: Any, candidates: Sequence[Mapping[str, Any]]
                     ) -> Mapping[str, Any]:
    """The question an output was asked and scored on: its id's candidates whose text is the one the agent was given
    and whose gold has the initial score's row count; every survivor must give the same gold."""
    if not isinstance(n_gold, int) or isinstance(n_gold, bool):
        raise PackageError(f"output {rid}: its initial score names no gold row count (n_gold_rows)")
    kept = [q for q in candidates if q.get("question") == text and len(q.get("gold_rows") or ()) == n_gold]
    if not kept:
        raise PackageError(f"output {rid}: no question {qid} among --questions has its text and gold row count")
    if len({json.dumps([q.get("gold_sparql"), q.get("gold_rows")], sort_keys=True) for q in kept}) > 1:
        raise PackageError(f"output {rid}: the question files give question {qid} two different golds")
    return kept[0]


def load_sample(path: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    header = doc.get("header") or {}
    if header.get("schema") != au.AUDIT_SCHEMA:
        raise PackageError(f"the sample is not an {au.AUDIT_SCHEMA} file")
    rids = [it["run_id"] for it in doc.get("items") or ()]
    if not rids:
        raise PackageError("the sample holds no items")
    if len(set(rids)) != len(rids):
        raise PackageError("the sample names an output twice")
    return doc


# --- one item -----------------------------------------------------------------------------------------------------

def _check_record(item: Mapping[str, Any], rec: Mapping[str, Any], record_sha256: str, split: str) -> None:
    rid = item["run_id"]
    seen = {"graph": (rec.get("store") or {}).get("name"), "model": rec.get("model"),
            "question_id": rec.get("question_id"), "base_question_id": rec.get("base_question_id"),
            "split": rec.get("split")}
    want = {"graph": item["graph"], "model": item["model"], "question_id": item["question_id"],
            "base_question_id": item["base_question_id"], "split": split}
    if bad := sorted(k for k in want if seen[k] != want[k]):
        raise PackageError(f"output {rid}: its run record disagrees with the sample on {', '.join(bad)}")
    if item["kind"] == "C":
        prompt = rec.get("prompt") or {}
        if record_sha256 != item.get("record_sha256"):
            raise PackageError(f"output {rid}: its run record is not the one the sample linked (record_sha256)")
        if prompt.get("condition") != item["cohort"] or prompt.get("ontology_block_sha256") != item.get(
                "ontology_block_sha256"):
            raise PackageError(f"output {rid}: its run record's prompt is not the sample's link")


def _check_statuses(rid: str, kind: str, arms: Mapping[str, Mapping[str, Any]], practice_only: bool = False
                    ) -> Mapping[str, Any] | None:
    """The curated row, after the kind's status rules (module docstring; a practice-only C needs only the row)."""
    if bad := sorted(a for a, r in arms.items() if r.get("status") not in LETTER):
        raise PackageError(f"output {rid}: matrix row of arm {bad[0]} has no known status")
    curated = arms.get(CURATED_ARM)
    if kind in ("B", "C") and curated is None:
        raise PackageError(f"output {rid}: kind {kind} needs the curated arm's primary row")
    if kind == "B":
        ok = curated["status"] == FLAGGED
    elif kind == "C":
        ok = practice_only or curated["status"] in PASSES
    else:
        ok = not any(r["status"] == FLAGGED for a, r in arms.items() if au.counts_for_residue(a))
    if not ok:
        raise PackageError(f"output {rid}: the matrix rows' statuses do not fit kind {kind}")
    return curated


def render_item(item: Mapping[str, Any], rec: Mapping[str, Any], record_sha256: str,
                arms: Mapping[str, Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]], split: str, *,
                private: bool, allow_family: Iterable[str] = ()) -> dict[str, Any]:
    """The item's rendering, built from ``SHOWN_BY_KIND`` alone (no item id yet), after every check."""
    rid, kind = item["run_id"], item["kind"]
    allow_family = set(allow_family)
    practice_only = bool(item.get("practice_only"))
    if kind == "C" and not practice_only and item["cohort"] != au.CURATED_COHORT:
        raise PackageError(f"output {rid}: kind C is the curated prompt's cohort, not {item['cohort']!r}")
    _check_record(item, rec, record_sha256, split)
    outcome, final_rows, score = initial_of(rec)
    if outcome.get("status") != "submitted":
        raise PackageError(f"output {rid}: its initial output is not a submission")
    correct = kind == "B"
    if (score or {}).get("exact") is not correct:
        want = "correct" if correct else "wrong"
        raise PackageError(f"output {rid}: its initial score is not kind {kind}'s ({want})")
    if any(r.get("correct") is not correct for r in arms.values()):
        raise PackageError(f"output {rid}: a matrix row's correctness is not kind {kind}'s")
    curated = _check_statuses(rid, kind, arms, practice_only)
    row = curated or next(iter(arms.values()))
    upto = (rec.get("checker") or {}).get("initial", {}).get("submission_index")
    output, cq = checked_output(rec["tool_calls"], final_rows, rules_of(score), upto=upto,
                                row_cap=(rec.get("store") or {}).get("row_cap"))
    query = output.query if output is not None else None
    if (query, cq) != (row.get("checked_query"), row.get("checked_query_status")):
        raise PackageError(f"output {rid}: rule 22's checked query recomputed from the record is not the matrix's")
    if cq not in QUERY_NOTES:
        raise PackageError(f"output {rid}: unknown checked-query status")
    q = resolve_question(rid, item["question_id"], question_text(rec), (score or {}).get("n_gold_rows"), candidates)
    shown: dict[str, Any] = {
        "question": q["question"], "gold_query": q["gold_sparql"], "gold_answer": list(q["gold_rows"]),
        "agent_query": query, "agent_query_note": QUERY_NOTES[cq], "agent_answer": list(final_rows or ()),
    }
    agent_text = json.dumps([query, list(final_rows or ())], ensure_ascii=False)
    if named := [w for w in FAMILY_WORDS if _word(w, agent_text) and w not in allow_family]:
        raise PackageError(f"output {rid}: the agent's final query or answer names a model family ({named[0]}); "
                           f"the protocol's owner rules, and --allow-family-word {rid}:{named[0]} shows it")
    if kind == "B":
        if not curated.get("fired"):
            raise PackageError(f"output {rid}: the curated row is flagged but has no message lines")
        shown["flag_message"] = render_lines(list(curated["fired"]))
    if kind == "C":
        try:  # a public item's block comes from the repository's store only (as the sampler links it)
            shown["ontology_text"] = prompts.load_block(item["ontology_block_sha256"],
                                                        None if private else prompts.BLOCK_DIR)
        except (OSError, ValueError) as e:
            raise PackageError(f"output {rid}: its ontology block is not in the block store with bytes that match "
                               f"its sha256 ({type(e).__name__})") from None
    return {"form": kind, **shown}


def _word(word: str, text: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z])", text.lower()) is not None


# --- shuffle, ids, rendering --------------------------------------------------------------------------------------

def _rank(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


def assign_ids(run_ids: Iterable[str], seed: str, prefix: str) -> dict[str, str]:
    """run id -> item id: numbered in the order of sha256("<seed>|id|<run id>")."""
    ranked = sorted(run_ids, key=lambda r: (_rank(seed, "id", r), r))
    return {r: f"{prefix}{i:03d}" for i, r in enumerate(ranked, 1)}


def annotator_order(item_ids: Iterable[str], seed: str, annotator: str) -> list[str]:
    return sorted(item_ids, key=lambda i: (_rank(seed, "order", annotator, i), i))


def term_text(cell: Any) -> str:
    try:
        (term,) = row_from_json([cell])
    except (TermError, ValueError, TypeError):
        return json.dumps(cell, ensure_ascii=False)
    if term is None:
        return "(unbound)"
    if isinstance(term, IRI):
        return f"<{term.value}>"
    if isinstance(term, BNode):
        return f"_:{term.label}"
    assert isinstance(term, Literal)
    text = json.dumps(term.lexical, ensure_ascii=False)
    return text + (f"@{term.lang}" if term.lang else f"^^<{term.datatype}>" if term.datatype else "")


def rows_text(rows: Sequence[Any]) -> str:
    lines = ["  |  ".join(term_text(c) for c in (r if isinstance(r, list) else [r])) for r in rows[:MAX_ROWS_SHOWN]]
    if len(rows) > MAX_ROWS_SHOWN:
        lines.append(f"... and {len(rows) - MAX_ROWS_SHOWN} more rows (all of them in items.jsonl)")
    return "\n".join(lines) if lines else "(no rows)"


def _fence(text: str, lang: str = "") -> str:
    run, longest = 0, 0
    for ch in text:
        run = run + 1 if ch == "`" else 0
        longest = max(longest, run)
    f = "`" * max(3, longest + 1)
    return f"{f}{lang}\n{text}\n{f}"


def _n_rows(n: int) -> str:
    return f"{n} row{'' if n == 1 else 's'}"


WITNESS_NOTE = "Form C asks for the witness: quote the statement the wrong answer follows from."


def item_markdown(r: Mapping[str, Any]) -> str:
    parts = [f"## {r['item_id']} ({FORM_TITLES[r['form']].split(':')[0]})", "",
             "### Question", _fence(r["question"], "text"), "",
             "### Gold query", _fence(r["gold_query"], "sparql"), "",
             f"### Gold answer ({_n_rows(len(r['gold_answer']))})", _fence(rows_text(r["gold_answer"])), "",
             "### The agent's final query"]
    if r["agent_query_note"]:
        parts.append(f"*{r['agent_query_note']}*")
    if r["agent_query"] is not None:
        parts.append(_fence(r["agent_query"], "sparql"))
    parts += ["", f"### The agent's answer ({_n_rows(len(r['agent_answer']))})", _fence(rows_text(r["agent_answer"]))]
    if "flag_message" in r:
        parts += ["", "### The checker's message", _fence(r["flag_message"], "text")]
    if "ontology_text" in r:
        parts += ["", "### The ontology text in the agent's prompt", _fence(r["ontology_text"], "text"),
                  "", f"*{WITNESS_NOTE}*"]
    return "\n".join(parts) + "\n"


def _codebook_markdown() -> str:
    lines = []
    for kind in au.KINDS:
        lines += [f"### {FORM_TITLES[kind]}", ""]
        for f in CODEBOOK[kind]:
            labels = ", ".join(f"`{x}`" for x in f.labels) if f.labels else "free text"
            lines.append(f"- **`{f.column}`**: {f.question} ({labels})"
                         + (f"; `other` needs a short note in `{f.note}`" if f.note else ""))
        lines.append("")
    return "\n".join(lines)


GUIDE_DEFINITIONS = """\
**Form A**
1. **Error type** (`error_type`):
   - `aggregation`: count against count-distinct, min against max, a wrong grouping;
   - `constant`: right type, wrong entity;
   - `filter_scope`: a condition missing or extra, or the wrong graph or scope;
   - `path`: the wrong relation path, or a missing or extra hop;
   - `shape`: wrong or missing columns, duplicates;
   - `gold_or_question`: the gold is wrong or the question is ambiguous;
   - `scorer`: the answer is equivalent but scored wrong;
   - `other`, with a short note (`error_type_note`).
2. **Structurally detectable?** (`structurally_detectable`: `yes`, `no` or `unsure`): could a vocabulary, class,
   domain/range or disjointness check over a **correct** schema flag this query in principle?

**Form B**
1. **Cause** (`cause`):
   - `undeclared_term`: the schema lacks a term the data uses;
   - `closed_world`: a membership the data doesn't assert;
   - `data_clash`: the data logically contradicts the schema;
   - `schema_too_narrow`: the axiom contradicts sensible use;
   - `checker_defect`: a rule misapplied;
   - `other`, with a note (`cause_note`).
2. **Was the agent's answer really correct?** (`answer_really_correct`: `yes`, `no` or `unsure`). This catches scorer
   errors.

**Form C**
1. **Does the wrong answer follow from a statement in the ontology shown in the prompt?** (`follows_from_ontology`:
   `yes`, `no` or `unsure`). If `yes`, quote the statement (`witness`); a `yes` without a quoted witness is coded
   `unsure`.
2. **If yes, is that statement wrong or misleading for this data?** (`statement_wrong_or_misleading`: `yes`, `no` or
   `unsure`)
"""


def guide_markdown(annotator: str, n_items: int, practice: bool, forms: Mapping[str, int]) -> str:
    title = "Audit annotation: PRACTICE package" if practice else "Audit annotation package"
    practice_note = ("These items come from development runs, for learning the codebook before the real package. "
                     "Practice codes are never analysed.\n\n" if practice else "")
    counts = ", ".join(f"Form {k}: {forms.get(k, 0)}" for k in au.KINDS)
    return f"""# {title} ({annotator})

{practice_note}You have {n_items} items ({counts}), in your own order in `items.md` (the same items, every row of
each answer, in `items.jsonl`). Code each item in a copy of `sheet.csv`, one row per item: fill the columns of the
item's form and leave the "-" columns alone. Code on your own: don't discuss items with the other annotator until both
sheets are done. Disagreements are settled afterwards by discussion.

## What each item shows

- the question, as the agent was given it;
- the gold query and the gold answer;
- the agent's final query and answer. The final query is the last query the agent ran whose result contained its
  submitted answer under the registered selection rule; where none did, or the agent ran no query, the item says so.
- **Form B** also shows the message a checker produced for that query.
- **Form C** also shows the ontology text that was in the agent's prompt, and asks you to quote, as the witness, the
  statement the wrong answer follows from.

The items are shuffled. Each item has an id (such as `{"P" if practice else "T"}001`); use it to find the item's row in
the sheet.

## Codebook (one primary code per field)

{GUIDE_DEFINITIONS}
## The sheet's columns

{_codebook_markdown()}
`comment` is free text for anything else.
"""


def sheet_csv(order: Sequence[str], forms: Mapping[str, str]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(SHEET_COLUMNS)
    for n, iid in enumerate(order, 1):
        own = set(FORM_COLUMNS[forms[iid]])
        w.writerow([n, iid, forms[iid], *("" if c in own else NA for c in SHEET_COLUMNS[3:-1]), ""])
    return buf.getvalue()


def _identifier_like(v: Any) -> bool:
    return isinstance(v, str) and len(v) >= 6 and any(c.isdigit() or c in "-_./:" for c in v)


def blocked_strings(items: Sequence[Mapping[str, Any]], record_sha256s: Iterable[str] = (),
                    rows: Iterable[Mapping[str, Any]] = ()) -> dict[str, str]:
    """Every identifier-like value an annotator must not see -> the field it comes from: the sample's entries, the
    records' hashes and the matrix rows' check ids, reasons, sources and arms."""
    out: dict[str, str] = {}

    def add(v: Any, field: str) -> None:
        if _identifier_like(v):
            out[v] = field

    for it in items:
        for k in ("run_id", "question_id", "base_question_id", "model", "model_group", "cohort", "cell",
                  "record_sha256", "prompt_id", "prompt_sha256", "ontology_block_sha256"):
            add(it.get(k), k)
        model = it.get("model")
        if isinstance(model, str) and "/" in model:
            add(model.rsplit("/", 1)[1], "model")
    for sha in record_sha256s:
        add(sha, "record_sha256")
    for r in rows:
        add(r.get("arm"), "arm")
        add(r.get("reason"), "reason")
        for src in r.get("per_source") or {}:
            add(src, "per_source")
        for fc in r.get("fired_checks") or ():
            if isinstance(fc, Mapping):
                for k in ("check_id", "reason", "source"):
                    add(fc.get(k), "fired_checks")
            else:
                add(fc, "fired_checks")
    out.update({s: "status" for s in PASSES})  # "flagged" is a plain word a query may hold
    return out


def leaks(text: str, blocked: Mapping[str, str]) -> list[str]:
    return sorted({field for s, field in blocked.items() if s in text})


# --- the package --------------------------------------------------------------------------------------------------

def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _dump(doc: Any) -> str:
    return json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def build(sample: Mapping[str, Any], records: Mapping[str, tuple[dict[str, Any], str]],
          rows: Mapping[str, Mapping[str, Mapping[str, Any]]], questions: Mapping[str, Sequence[Mapping[str, Any]]],
          *, seed: str, annotators: Sequence[str] = ANNOTATORS, practice: bool = False, private: bool = False,
          inputs: Mapping[str, Any] | None = None, allow_family: Mapping[str, Iterable[str]] | None = None
          ) -> tuple[dict[str, str], str]:
    """The package's files (relative path -> text) and the key's text, all in memory. ``rows``: each output's primary
    matrix rows by arm (``matrix_rows``); ``allow_family``: run id -> the family words the protocol's owner let
    through for that output."""
    header = sample["header"]
    split = header.get("split")
    if bool(header.get("private")) != private:
        raise PackageError("a private sample is built only with --private, and --private builds only a private one")
    if (split == "dev") != practice:
        raise PackageError("a dev sample is built only as a practice package (--practice), a test sample never")
    if not annotators or len(set(annotators)) != len(annotators) or any(
            not a or not all(c.isalnum() or c in "-_" for c in a) for a in annotators):
        raise PackageError("annotators need distinct names of letters, digits, '-' and '_'")
    items = list(sample["items"])
    if not private and (bad := [it for it in items if it["graph"] in au.PRIVATE_GRAPHS]):
        raise PackageError(f"a public package never holds a private graph's item ({len(bad)} given)")
    if private and any(it["graph"] not in au.PRIVATE_GRAPHS for it in items):
        raise PackageError("a private package holds the private graphs' items only")
    if not practice and any(it.get("practice_only") for it in items):
        raise PackageError("a practice-only item is built only into a practice package")
    allow_family = dict(allow_family or {})
    if unknown := sorted(set(allow_family) - {it["run_id"] for it in items}):
        raise PackageError(f"--allow-family-word names output {unknown[0]}, which is not in the sample")
    rendered: dict[str, dict[str, Any]] = {}
    for it in items:
        rid = it["run_id"]
        rec, rsha = records[rid]
        rendered[rid] = render_item(it, rec, rsha, rows[rid], questions.get(it["question_id"], ()), split,
                                    private=private, allow_family=allow_family.get(rid, ()))
    ids = assign_ids(rendered, seed, "P" if practice else "T")
    by_id = {ids[rid]: {"item_id": ids[rid], **r} for rid, r in rendered.items()}
    forms = {iid: r["form"] for iid, r in by_id.items()}
    form_counts = dict(sorted(Counter(forms.values()).items()))
    blocked = blocked_strings(items, (records[it["run_id"]][1] for it in items),
                              (r for it in items for r in rows[it["run_id"]].values()))
    files: dict[str, str] = {}
    orders: dict[str, list[str]] = {}
    for a in annotators:
        order = annotator_order(by_id, seed, a)
        orders[a] = order
        base = f"annotators/{a}"
        files[f"{base}/README.md"] = guide_markdown(a, len(order), practice, form_counts)
        files[f"{base}/items.md"] = "\n".join(item_markdown(by_id[i]) for i in order)
        files[f"{base}/items.jsonl"] = "".join(json.dumps(by_id[i], ensure_ascii=False, sort_keys=True) + "\n"
                                               for i in order)
        files[f"{base}/sheet.csv"] = sheet_csv(order, forms)
    files["HANDLING.md"] = handling_markdown(list(annotators), practice, private)
    for name, text in files.items():
        if found := leaks(text, blocked):
            where = "an annotator file" if private else name
            raise PackageError(f"a blocked field ({', '.join(found)}) appears in {where}")
    key_items = {}
    for it in items:
        rid = it["run_id"]
        key_items[ids[rid]] = {"sample_item": it, "record_sha256": records[rid][1]}
    key = {"schema": KEY_SCHEMA, "protocol": PROTOCOL, "private": private, "practice": practice, "split": split,
           "shuffle_seed": seed, "id_rule": "items numbered in the order of sha256('<seed>|id|<run id>')",
           "order_rule": "per annotator, ids by sha256('<seed>|order|<annotator>|<item id>')",
           "orders": orders, "items": dict(sorted(key_items.items())),
           "files_sha256": {n: _sha256_bytes(t.encode("utf-8")) for n, t in sorted(files.items())}}
    key_text = _dump(key)
    extra = sorted(ids[it["run_id"]] for it in items if it.get("practice_only"))
    strict = dict(sorted(Counter(forms[ids[it["run_id"]]] for it in items if not it.get("practice_only")).items()))
    manifest = {"schema": PACKAGE_SCHEMA, "protocol": PROTOCOL, "private": private, "practice": practice,
                "split": split, "annotators": list(annotators), "items": len(by_id), "items_by_form": form_counts,
                "strict_items_by_form": strict,
                "practice_only": {"items": extra, "rule": header.get("practice_only_rule")} if extra else None,
                "family_words_allowed": {ids[r]: sorted(set(w)) for r, w in sorted(allow_family.items())},
                "shown": {k: list(v) for k, v in SHOWN_BY_KIND.items()}, "not_shown": list(NOT_SHOWN),
                "shuffle_seed_sha256": _sha256_bytes(seed.encode("utf-8")),
                "key_sha256": _sha256_bytes(key_text.encode("utf-8")),
                "files_sha256": key["files_sha256"], "inputs": dict(inputs or {}),
                "builder_sha256": _file_sha256(Path(__file__)),
                "handling": "annotators get annotators/<name>/ only; the sample and the key stay away from them until "
                            "their sheets are locked (the sample names each item's question and model)"}
    files["manifest.json"] = _dump(manifest)
    return files, key_text


def handling_markdown(annotators: Sequence[str], practice: bool, private: bool) -> str:
    kind = "practice " if practice else ""
    where = ("It is a private package: keep it in mode-600 files under owner-only directories, never in git, never "
             "in a log.\n\n" if private else "")
    folders = "\n".join(f"- `annotators/{a}/`" for a in annotators)
    release = ("The key goes to the annotators only after the practice round." if practice else
               "The key is opened only after both annotators' sheets are locked.")
    return f"""# Handling this {kind}audit package (for the lead, not for annotators)

{where}Give each annotator only their own folder:

{folders}

Withhold until every annotator's sheet is locked: the drawn sample (it names each item's question, model and cohort,
and a question's text identifies its item), the key (`--key-out`), the inputs the package was built from, this file
and `manifest.json`. The manifest commits to the sample by its sha256 (`inputs.sample_sha256`) and to the key
(`key_sha256`).

Annotators do not open the experiments repository, its run records (dev records included) or the compute environment
until their coding ends: the records pair each question with its model, and the code redraws the sample. {release}

The real audit package is built only in the private environment and delivered to the annotators off GitHub:
nothing that pairs items with models or arms (the sample, the key, the inputs, the records) leaves the cluster before
both annotators' sheets are in.

Annotators fill a copy of their `sheet.csv`. Check a filled sheet against this package:

    uv run python -m harness.analysis.audit_package --check-sheet FILLED.csv --package <this directory> \\
        --annotator <name> [--complete]
"""


def _closed_dir(path: Path) -> bool:
    return path.is_dir() and not (path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO))


def check_paths(out: Path, key_out: Path, private: bool) -> None:
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise PackageError("--out exists and is not an empty directory")
    if key_out.exists():
        raise PackageError("--key-out exists")
    if key_out.is_relative_to(out):
        raise PackageError("--key-out must be outside --out: the key is never shown to annotators")
    if private:
        if out.is_relative_to(ROOT) or key_out.is_relative_to(ROOT):
            raise PackageError("a private package and its key must be outside the repository")
        if not _closed_dir(out.parent) or not _closed_dir(key_out.parent):
            raise PackageError("a private package and its key go under directories open to their owner only")
        if out.exists() and not _closed_dir(out):
            raise PackageError("an existing --out of a private package must be open to its owner only")


def write_package(out: Path, key_out: Path, files: Mapping[str, str], key_text: str, private: bool) -> None:
    """Stage the package beside ``out`` and the key beside ``key_out``, then move both into place; on any failure
    remove whatever was staged or moved, so a refusal leaves nothing."""
    file_mode, dir_mode = (0o600, 0o700) if private else (0o644, 0o755)
    out.parent.mkdir(parents=True, exist_ok=True)
    key_out.parent.mkdir(parents=True, exist_ok=True)
    existed = out.exists()
    staging: Path | None = None
    key_tmp: str | None = None
    moved = False
    try:
        staging = Path(tempfile.mkdtemp(prefix=".audit-package-", dir=out.parent))
        fd, key_tmp = tempfile.mkstemp(prefix=".audit-key-", dir=key_out.parent)
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(key_text)
        for name, text in sorted(files.items()):
            path = staging / name
            for d in reversed(path.relative_to(staging).parents[:-1]):
                (staging / d).mkdir(mode=dir_mode, exist_ok=True)
                os.chmod(staging / d, dir_mode)
            wfd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, file_mode)
            os.fchmod(wfd, file_mode)
            with os.fdopen(wfd, "w", encoding="utf-8") as f:
                f.write(text)
        os.chmod(staging, dir_mode)
        os.rename(staging, out)  # onto a missing or empty directory (check_paths)
        moved = True
        os.link(key_tmp, key_out)  # never over an existing file
    except BaseException:
        if moved:
            shutil.rmtree(out, ignore_errors=True)
            if existed:
                out.mkdir(mode=dir_mode)
        raise
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        if key_tmp is not None:
            Path(key_tmp).unlink(missing_ok=True)


# --- the annotators' sheets ---------------------------------------------------------------------------------------

def _read_sheet(path: Path) -> tuple[list[str] | None, list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def check_sheet(path: Path, package: Path, annotator: str, complete: bool = False) -> list[str]:
    """Problems of a filled sheet against the package's own sheet for ``annotator`` (whose bytes must be the ones
    the package's manifest names): another set or order of items or forms, unknown labels, a column of another form
    filled, an ``other`` with no note, a C ``yes`` with no witness (coded ``unsure`` by the codebook); with
    ``complete``, an empty code of the item's form."""
    name = f"annotators/{annotator}/sheet.csv"
    against = package / name
    try:
        listed = json.loads((package / "manifest.json").read_text(encoding="utf-8"))["files_sha256"][name]
    except (OSError, ValueError, KeyError):
        return [f"the package has no manifest naming {name}"]
    if not against.is_file() or _file_sha256(against) != listed:
        return [f"the package's {name} is not the one its manifest names"]
    if path.resolve() == against.resolve():
        return ["the filled sheet is the package's own: fill a copy"]
    columns, rows = _read_sheet(path)
    want_columns, want = _read_sheet(against)
    if columns != SHEET_COLUMNS or want_columns != SHEET_COLUMNS:
        return ["the sheet's columns are not the package's"]
    problems = []
    if [(r["order"], r["item_id"], r["form"]) for r in rows] != [(r["order"], r["item_id"], r["form"]) for r in want]:
        problems.append(f"the sheet's items are not the package's: {len(rows)} rows against {len(want)}, or another "
                        "id, form or order")
    for row in rows:
        iid, form = row["item_id"], row["form"]
        if form not in CODEBOOK:
            problems.append(f"{iid}: unknown form {form!r}")
            continue
        own = set(FORM_COLUMNS[form])
        for c in SHEET_COLUMNS[3:-1]:
            if c not in own and row[c] not in ("", NA):
                problems.append(f"{iid}: {c} is not a Form {form} column")
        for fld in CODEBOOK[form]:
            v = row[fld.column].strip()
            if fld.labels is not None and v and v not in fld.labels:
                problems.append(f"{iid}: {fld.column} {v!r} is not a codebook label")
            if fld.note and v == "other" and not row[fld.note].strip():
                problems.append(f"{iid}: {fld.column} other needs a note in {fld.note}")
            if complete and not v and fld.labels is not None and not (
                    fld.column == "statement_wrong_or_misleading" and row["follows_from_ontology"].strip() != "yes"):
                problems.append(f"{iid}: {fld.column} is empty")
        if form == "C" and row["follows_from_ontology"].strip() == "yes" and not row["witness"].strip():
            problems.append(f"{iid}: a yes without a quoted witness is coded unsure")
        if form == "C" and row["follows_from_ontology"].strip() != "yes" and row[
                "statement_wrong_or_misleading"].strip():
            problems.append(f"{iid}: statement_wrong_or_misleading is coded only after a yes to follows_from_ontology")
    return problems


# --- the command line ---------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the audit's annotation package from a drawn sample.")
    ap.add_argument("--sample", type=Path, help="the drawn sample (h4-audit-sample/2), public or private")
    ap.add_argument("--records", type=Path, nargs="+", default=[], help="run records: OUT directories or archives")
    ap.add_argument("--matrix", type=Path, nargs="+", default=[], help="the offline replay matrices")
    ap.add_argument("--questions", type=Path, nargs="+", default=[], help="question sets (JSON lines)")
    ap.add_argument("--out", type=Path, help="the package directory (new or empty)")
    ap.add_argument("--key-out", type=Path, help="the sealed key (a new file, outside --out)")
    ap.add_argument("--annotators", nargs="+", default=list(ANNOTATORS))
    ap.add_argument("--shuffle-seed", help="rebuild with a key's seed (32 hex digits); default: a fresh one")
    ap.add_argument("--practice", action="store_true", help="a practice package from a dev sample")
    ap.add_argument("--private", action="store_true", help="the private graphs' package (mode 600, outside the repo)")
    ap.add_argument("--allow-family-word", action="append", default=[], metavar="RUN_ID:WORD",
                    help="let a model family's name through in that output's query or answer (the protocol owner's "
                         "ruling; repeatable)")
    ap.add_argument("--check-sheet", type=Path, help="check a filled sheet instead of building")
    ap.add_argument("--package", type=Path, help="with --check-sheet: the package directory")
    ap.add_argument("--annotator", help="with --check-sheet: whose sheet")
    ap.add_argument("--complete", action="store_true", help="with --check-sheet: every code must be filled")
    args = ap.parse_args(argv)
    if args.check_sheet:
        if not (args.package and args.annotator):
            ap.error("--check-sheet needs --package and --annotator")
        problems = check_sheet(args.check_sheet, args.package, args.annotator, args.complete)
        for p in problems:
            print(p)
        print(f"{len(problems)} problem(s)")
        return 0 if not problems else 1
    if not (args.sample and args.out and args.key_out and args.records and args.matrix and args.questions):
        ap.error("a build needs --sample, --records, --matrix, --questions, --out and --key-out")
    if args.shuffle_seed is not None and not SEED_RE.fullmatch(args.shuffle_seed):
        ap.error("--shuffle-seed takes a key's seed: 32 lowercase hex digits")
    allow_family: dict[str, set[str]] = defaultdict(set)
    for spec in args.allow_family_word:
        rid, _, word = spec.rpartition(":")
        if not rid or word not in FAMILY_WORDS:
            ap.error(f"--allow-family-word takes RUN_ID:WORD, WORD one of {', '.join(FAMILY_WORDS)}")
        allow_family[rid].add(word)
    out, key_out = args.out.resolve(), args.key_out.resolve()
    ids: list[str] = []  # masked in a private build's refusal
    try:
        check_paths(out, key_out, args.private)
        sample = load_sample(args.sample)
        rids = [it["run_id"] for it in sample["items"]]
        ids = [v for it in sample["items"] for k in ("run_id", "question_id", "base_question_id")
               if isinstance(v := it.get(k), str) and v]
        if missing := [i for i, p in enumerate(args.records, 1) if not p.exists()]:
            raise PackageError(f"--records path {missing[0]} of {len(args.records)} does not exist")
        records = records_of(load_raw_records(args.records), rids)
        rows = matrix_rows(args.matrix, rids)
        questions = question_sets(args.questions)
        inputs = {"sample_sha256": _file_sha256(args.sample),
                  "questions_sha256": sorted(_file_sha256(p) for p in args.questions),
                  "matrices_sha256": sorted(_file_sha256(p) for p in args.matrix)}
        files, key_text = build(sample, records, rows, questions, seed=args.shuffle_seed or secrets.token_hex(16),
                                annotators=args.annotators, practice=args.practice, private=args.private,
                                inputs=inputs, allow_family=allow_family)
        write_package(out, key_out, files, key_text, args.private)
    except (PackageError, OSError, ValueError, KeyError) as e:
        msg = str(e)
        if args.private:  # the field and the reason, never an item's ids
            for v in sorted(set(ids), key=len, reverse=True):
                msg = msg.replace(v, "<id>")
        print(f"audit package refused: {msg}", file=sys.stderr)
        return 2
    if args.private:
        print("wrote the private package and its key (mode 600; nothing of it printed)")
        return 0
    m = json.loads(files["manifest.json"])
    print(f"wrote {m['items']} items {m['items_by_form']} for {', '.join(m['annotators'])} to {out}; "
          f"key {key_out} (sha256 {m['key_sha256'][:16]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
