"""The post-run audit's analysis (S3.1; pre-registration §5 "Audits", H4 and H6; ``paper/audit-protocol.md``): Cohen's
kappa between the two annotators, and the adjudicated codes' counts with shares, from the annotators' two filled
sheets, the adjudicated codes and the sealed key.

    uv run python -m harness.analysis.audit_analysis --package PACKAGE_DIR \\
        --sheet annotator-1=FILLED1.csv --sheet annotator-2=FILLED2.csv --adjudicated ADJUDICATED.csv \\
        --key KEY.json --out OUT_DIR [--flags FLAGS.csv] [--print]

Exit 0: written; 2: refused, and nothing is written. Files go to a new or empty ``--out`` (directory 700, files 600,
outside this repository: analysis outputs are the owner's); a private package's also need an ``--out`` parent open to
its owner only, and nothing of it is printed. Without ``--print`` nothing of the results is printed either.

**The gate** (in this order; the key is opened last, only when every check before it has passed):

1. The package is a real one: ``manifest.json`` of ``audit_package.PACKAGE_SCHEMA``, not a practice package (practice
   codes are never analysed), with two distinct annotators; ``--sheet`` names exactly those two.
2. Each sheet is **locked**, which here means: ``audit_package.check_sheet(..., complete=True)`` finds no problem in it
   against the package's own sheet for that annotator (same items, forms and order, only the codebook's labels, every
   code of the item's form filled, a note for each ``other``, a witness for each ``yes`` of form C), and the file did
   not change while it was read. The sha256 of each sheet is written to the output: the lead compares it with
   the hash taken when the annotator handed the sheet in.
3. The adjudicated codes (format below) are complete and valid for the package's items and forms.
4. ``--flags`` (optional) is well formed.
5. Only now is the key read: its sha256 must be the manifest's ``key_sha256``; its seed's sha256 the manifest's; its
   items, their forms, the ids the seed gives the run ids, the annotators' orders and the files' hashes must agree with
   the package and the sheets. A mismatch refuses.

**The adjudicated codes** (``--adjudicated``, CSV, UTF-8): the same columns, in the same order, as an annotator's
``sheet.csv`` (``order,item_id,form,<A's columns>,<B's>,<C's>,comment``), one row per item in any order; ``order`` and
``comment`` are ignored. Fill the item's form's columns with the codes agreed by discussion (an item both annotators
coded alike keeps their code; an override of an agreed code is allowed but counted and listed in the output: a
field both annotators gave the same non-empty code, adjudicated differently, with ``statement_wrong_or_misleading``
not counted again when its parent ``follows_from_ontology`` is itself an override), put
"-" in the other forms' columns, and satisfy the same rules as a filled sheet.

**The flag table** (``--flags``, optional, CSV, UTF-8): columns ``run_id,source,rule``, one row per kind B item (the
key's run id of every B item, exactly once; no other): the source that flagged it ("curated") and the fired check's
rule id. The key does not carry the rule (nor the matrix's fired checks): whoever may read the matrix produces this
table from it, one primary rule per item. Without it the by-rule table is absent and every B item's source kind is
"curated", as the draw makes it (audit_sample: B is the curated source's flags).

**Agreement.** Cohen's kappa is computed per codebook field with labels (``error_type``, ``structurally_detectable``;
``cause``, ``answer_really_correct``; ``follows_from_ontology``, ``statement_wrong_or_misleading``), grouped by form,
the first field of each form being its primary code; free text (``witness``, the notes, ``comment``) has none. Each
field also gets a one-versus-rest kappa per label. Over the items both annotators coded: all of a form's items, except
``statement_wrong_or_misleading``, coded only after a yes to ``follows_from_ontology``, whose kappa is over the items
both answered yes to (the n is reported, as are the items only one of them coded). kappa = (p_o - p_e) / (1 - p_e)
from the annotators' two label sequences; it is null, with a reason, when p_e = 1 (both used one and the same label
for every item) or no item was coded by both. The protocol asks for the coefficient only: no interval is given.

**Counts and shares.** Every count carries its denominator ``n`` and its share (null when ``n`` = 0), with an exact
Clopper-Pearson 95% interval over items (``stats.clopper_pearson``). Shares are over the sampled items, unweighted
by cell: the sample is stratified with equal quotas (8 per cell), so a share is not an estimate of the population's.
Counts are of the adjudicated codes. Each table names its denominators:

* H4 residue (form A): ``error_type`` and ``structurally_detectable`` over ``all_items`` (every A item) and over
  ``misses_only`` (A items not coded ``gold_or_question`` or ``scorer``: a wrong gold, an ambiguous question or a
  scorer error is not an agent's miss). The task's misses by error kind are ``aggregation``, ``constant`` (wrong
  constant), ``filter_scope`` (filter or scope), ``path``, ``shape`` (answer shape) and ``other``; by graph and by
  model group as well.
* False alarms (form B): ``cause`` over ``all_items`` (every B item) and over ``answer_really_correct_yes`` (B items
  whose adjudicated ``answer_really_correct`` is ``yes``: a flag on an answer that was not really correct is no false
  alarm), by source kind (and, with a flag table, by rule within a source kind), overall and by graph. The task's
  three causes are ``data_clash`` (the data breaks the axiom), ``closed_world`` (accepted under OWL's open-world
  reading) and ``undeclared_term`` (the schema lacks a term the data uses); ``schema_too_narrow``, ``checker_defect``
  and ``other`` stay separate rows.
* H6 shared premise (form C): ``follows_from_ontology`` over all C items; ``statement_wrong_or_misleading`` over the
  C items adjudicated ``yes`` on the first.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import stat
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from harness.analysis import audit_package as ap
from harness.analysis import audit_sample as au
from harness.analysis import stats

RESULT_SCHEMA = "h4-audit-analysis/1"
ROOT = Path(__file__).resolve().parents[2]
FORM_NAMES = {"A": "residue", "B": "false flag", "C": "shared premise"}
NOT_MISSES = ("gold_or_question", "scorer")  # form A codes that are no error of the agent's
DEFAULT_SOURCE = "curated"  # a form B item is the curated source's flag (audit_sample.in_kind)
FLAG_COLUMNS = ["run_id", "source", "rule"]
LABEL_DESC = {
    "error_type": {"aggregation": "aggregation", "constant": "wrong constant", "filter_scope": "filter or scope",
                   "path": "path", "shape": "answer shape", "gold_or_question": "wrong gold or ambiguous question",
                   "scorer": "equivalent answer scored wrong", "other": "other"},
    "cause": {"undeclared_term": "the schema lacks a term the data uses",
              "closed_world": "accepted under OWL's open-world reading (a membership the data doesn't assert)",
              "data_clash": "the data breaks the axiom", "schema_too_narrow": "the axiom contradicts sensible use",
              "checker_defect": "a rule misapplied", "other": "other"},
}
FIELDS = {k: [f for f in ap.CODEBOOK[k] if f.labels is not None] for k in au.KINDS}
LABELS = {f.column: f.labels for k in au.KINDS for f in FIELDS[k]}
CONDITIONAL = "statement_wrong_or_misleading"  # coded only after a yes to follows_from_ontology


class AnalysisError(ValueError):
    """An input the analysis refuses. Messages name item ids and fields, never content."""


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _closed(path: Path) -> bool:
    return path.is_dir() and not (path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO))


# --- agreement ----------------------------------------------------------------------------------------------------

def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> dict[str, Any]:
    """Cohen's kappa of two annotators' labels for the same items: (p_o - p_e) / (1 - p_e), computed in integers as
    (n*agree - sum_l n_a(l) n_b(l)) / (n^2 - sum_l n_a(l) n_b(l)); null, with a reason, when p_e = 1 or n = 0."""
    if len(a) != len(b):
        raise AnalysisError("two label sequences of different lengths")
    n = len(a)
    agree = sum(x == y for x, y in zip(a, b, strict=True))
    ca, cb = Counter(a), Counter(b)
    pe_num = sum(ca[k] * cb[k] for k in ca.keys() & cb.keys())
    out: dict[str, Any] = {"n": n, "agree": agree, "observed_agreement": round(agree / n, 6) if n else None,
                           "expected_agreement": round(pe_num / n ** 2, 6) if n else None,
                           "kappa": None, "reason": None}
    if n == 0:
        out["reason"] = "no item was coded by both annotators"
    elif pe_num == n * n:
        out["reason"] = "undefined: both annotators used one and the same label for every item (p_e = 1)"
    else:
        out["kappa"] = round((n * agree - pe_num) / (n * n - pe_num), 6)
    return out


def agreement(forms: Mapping[str, str], c1: Mapping[str, Mapping[str, str]], c2: Mapping[str, Mapping[str, str]]
              ) -> dict[str, Any]:
    """Kappa per form and field (module docstring), from the two annotators' codes by item id."""
    out: dict[str, Any] = {}
    for kind in au.KINDS:
        ids = sorted(i for i, f in forms.items() if f == kind)
        fields: dict[str, Any] = {}
        for n, f in enumerate(FIELDS[kind]):
            both = [i for i in ids if c1[i][f.column] and c2[i][f.column]]
            a, b = [c1[i][f.column] for i in both], [c2[i][f.column] for i in both]
            row = cohen_kappa(a, b)
            row.update({"primary": n == 0, "items_in_form": len(ids), "coded_by_one_only": sum(
                bool(c1[i][f.column]) != bool(c2[i][f.column]) for i in ids),
                "disagreements": row["n"] - row["agree"],
                "per_label": {lab: {"annotator_1": sum(x == lab for x in a), "annotator_2": sum(x == lab for x in b),
                                    **{k: v for k, v in cohen_kappa([x == lab for x in a], [y == lab for y in b]
                                                                    ).items() if k in ("kappa", "reason")}}
                              for lab in f.labels}})
            fields[f.column] = row
        out[kind] = {"form": FORM_NAMES[kind], "items": len(ids), "fields": fields}
    return out


# --- counts with shares --------------------------------------------------------------------------------------------

def cell(k: int, n: int) -> dict[str, Any]:
    ci = stats.clopper_pearson(k, n)
    return {"count": k, "n": n, "share": round(k / n, 6) if n else None,
            "ci95": [round(ci[0], 6), round(ci[1], 6)] if ci else None}


def count_table(values: Sequence[str], labels: Sequence[str]) -> dict[str, Any]:
    c = Counter(values)
    if extra := sorted(set(c) - set(labels)):
        raise AnalysisError(f"a code outside the codebook reached the tables ({extra[0]!r})")
    return {"n": len(values), "labels": {lab: cell(c[lab], len(values)) for lab in labels}}


Subset = tuple[Callable[[Mapping[str, Any]], bool], Sequence[str]]


def views(rows: Sequence[Mapping[str, Any]], field: str, subsets: Mapping[str, Subset]) -> dict[str, Any]:
    return {name: count_table([r[field] for r in rows if keep(r)], labels) for name, (keep, labels) in subsets.items()}


def breakdown(rows: Sequence[Mapping[str, Any]], field: str, subsets: Mapping[str, Subset], *,
              model_groups: bool = False) -> dict[str, Any]:
    out = {"overall": views(rows, field, subsets),
           "by_graph": {g: views([r for r in rows if r["graph"] == g], field, subsets)
                        for g in sorted({r["graph"] for r in rows})}}
    if model_groups:
        out["by_model_group"] = {g: views([r for r in rows if r["model_group"] == g], field, subsets)
                                 for g in sorted({r["model_group"] for r in rows})}
    return out


def _all(_: Mapping[str, Any]) -> bool:
    return True


def by_rule(b_rows: Sequence[Mapping[str, Any]], subsets: Mapping[str, Subset], sources: Sequence[str]
            ) -> dict[str, Any]:
    """Cause tables by the fired check's rule within each source kind (the flag table's rule)."""
    out = {}
    for s in sources:
        own = [r for r in b_rows if r["source"] == s]
        out[s] = {rule: views([r for r in own if r["rule"] == rule], "cause", subsets)
                  for rule in sorted({r["rule"] for r in own})}
    return out


def tables(rows: Sequence[Mapping[str, Any]], flags: Mapping[str, tuple[str, str]] | None) -> dict[str, Any]:
    """H4, the false alarms and H6 from the adjudicated rows (each with its form, graph, model group and codes);
    ``flags``: B run id -> (source, rule) from the flag table, when given."""
    by = {k: [r for r in rows if r["form"] == k] for k in au.KINDS}
    err = LABELS["error_type"]
    miss = [x for x in err if x not in NOT_MISSES]
    a_sub: dict[str, Subset] = {"all_items": (_all, err),
                                "misses_only": (lambda r: r["error_type"] not in NOT_MISSES, miss)}
    yn_sub: dict[str, Subset] = {"all_items": (_all, ap.YNU),
                                 "misses_only": (lambda r: r["error_type"] not in NOT_MISSES, ap.YNU)}
    h4 = {"items": len(by["A"]),
          "denominators": {"all_items": "every form A item",
                           "misses_only": "form A items not coded " + " or ".join(NOT_MISSES)},
          "error_type": breakdown(by["A"], "error_type", a_sub, model_groups=True),
          "structurally_detectable": breakdown(by["A"], "structurally_detectable", yn_sub, model_groups=True),
          "other_notes": [{"item_id": r["item_id"], "note": r["error_type_note"]} for r in by["A"]
                          if r["error_type"] == "other"]}
    cause = LABELS["cause"]
    b_sub: dict[str, Subset] = {"all_items": (_all, cause),
                                "answer_really_correct_yes": (lambda r: r["answer_really_correct"] == "yes", cause)}
    b_rows = [{**r, "source": (flags or {}).get(r["run_id"], (DEFAULT_SOURCE, None))[0],
               "rule": (flags or {}).get(r["run_id"], (None, None))[1]} for r in by["B"]]
    sources = sorted({r["source"] for r in b_rows})
    fa: dict[str, Any] = {
        "items": len(b_rows),
        "denominators": {"all_items": "every form B item of the source kind",
                         "answer_really_correct_yes": "its items whose adjudicated answer_really_correct is yes"},
        "source_kind_basis": "the flag table" if flags else f"{DEFAULT_SOURCE!r} for every item (the draw: B is the "
                                                             "curated source's flags)",
        "by_source_kind": {s: {
            "cause": breakdown([r for r in b_rows if r["source"] == s], "cause", b_sub),
            "answer_really_correct": breakdown([r for r in b_rows if r["source"] == s], "answer_really_correct",
                                               {"all_items": (_all, ap.YNU)})} for s in sources},
        "by_rule": by_rule(b_rows, b_sub, sources) if flags else None,
        "other_notes": [{"item_id": r["item_id"], "note": r["cause_note"]} for r in by["B"] if r["cause"] == "other"]}
    yes = [r for r in by["C"] if r["follows_from_ontology"] == "yes"]
    h6 = {"items": len(by["C"]),
          "denominators": {"follows_from_ontology": "every form C item",
                           "statement_wrong_or_misleading": "form C items adjudicated yes on follows_from_ontology"},
          "follows_from_ontology": breakdown(by["C"], "follows_from_ontology", {"all_items": (_all, ap.YNU)}),
          "statement_wrong_or_misleading": breakdown(yes, "statement_wrong_or_misleading",
                                                     {"follows_yes": (_all, ap.YNU)})}
    return {"H4_residue": h4, "false_alarms": fa, "H6_shared_premise": h6}


# --- inputs -------------------------------------------------------------------------------------------------------

def _csv_rows(data: bytes, what: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""))
        rows = list(reader)
        return list(reader.fieldnames or []), rows
    except (UnicodeDecodeError, csv.Error) as e:
        raise AnalysisError(f"{what} is not a readable UTF-8 CSV ({type(e).__name__})") from None


def _read(path: Path, what: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as e:
        raise AnalysisError(f"{what} cannot be read ({type(e).__name__})") from None


def load_package(package: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(_read(package / "manifest.json", "the package's manifest.json"))
    except ValueError:
        raise AnalysisError("the package's manifest.json is not JSON") from None
    if not isinstance(manifest, dict) or manifest.get("schema") != ap.PACKAGE_SCHEMA:
        raise AnalysisError(f"the package's manifest is not an {ap.PACKAGE_SCHEMA} file")
    if manifest.get("practice"):
        raise AnalysisError("a practice package is never analysed: practice codes are not analysed")
    names = manifest.get("annotators")
    if not (isinstance(names, list) and len(names) == 2 and len(set(names)) == 2):
        raise AnalysisError("the package must name two distinct annotators")
    for k in ("key_sha256", "shuffle_seed_sha256", "files_sha256", "items_by_form", "items"):
        if k not in manifest:
            raise AnalysisError(f"the package's manifest lacks {k}")
    return manifest


def locked_sheet(path: Path, package: Path, annotator: str) -> tuple[dict[str, dict[str, str]], str]:
    """An annotator's codes by item id and the sheet's sha256, once the sheet passes ``check_sheet`` complete (the
    gate's second step); the file is hashed before the check and read again after it, and must not have changed."""
    before = _sha256(_read(path, f"{annotator}'s sheet"))
    try:
        problems = ap.check_sheet(path, package, annotator, complete=True)
    except Exception as e:  # a malformed file: refuse it, never go on
        raise AnalysisError(f"{annotator}'s sheet could not be checked ({type(e).__name__})") from None
    if problems:
        shown = "; ".join(problems[:5])
        raise AnalysisError(f"{annotator}'s sheet is not locked: {len(problems)} problem(s), first: {shown}")
    data = _read(path, f"{annotator}'s sheet")
    if _sha256(data) != before:
        raise AnalysisError(f"{annotator}'s sheet changed while it was read")
    _, rows = _csv_rows(data, f"{annotator}'s sheet")
    return _codes(rows), before


def _codes(rows: Sequence[Mapping[str, str]]) -> dict[str, dict[str, str]]:
    """Item id -> its form's codes (``form`` and every codebook column of that form, stripped; notes included)."""
    out = {}
    for r in rows:
        form = r["form"]
        out[r["item_id"]] = {"form": form, **{c: (r[c] or "").strip() for c in ap.FORM_COLUMNS[form]}}
    return out


def check_codes(rows: Sequence[Mapping[str, str]], want: Mapping[str, str]) -> list[str]:
    """Problems of the adjudicated codes' rows against the package's item ids and forms (``want``): another set of
    items, a duplicate, another form, a column of another form filled, an unknown label, an empty code, an
    ``other`` with no note, a form C ``yes`` with no witness, a second code after anything but a ``yes``."""
    problems, seen = [], set()
    for r in rows:
        iid, form = r.get("item_id") or "", r.get("form") or ""
        if iid in seen:
            problems.append(f"{iid}: two rows")
        seen.add(iid)
        if iid not in want:
            problems.append(f"{iid}: not an item of the package")
            continue
        if form != want[iid]:
            problems.append(f"{iid}: form {form!r} is not the package's {want[iid]!r}")
            continue
        own = set(ap.FORM_COLUMNS[form])
        for c in ap.SHEET_COLUMNS[3:-1]:
            if c not in own and (r.get(c) or "") not in ("", ap.NA):
                problems.append(f"{iid}: {c} is not a Form {form} column")
        val = {c: (r.get(c) or "").strip() for c in ap.FORM_COLUMNS[form]}
        for f in ap.CODEBOOK[form]:
            v = val[f.column]
            if f.labels is not None and v and v not in f.labels:
                problems.append(f"{iid}: {f.column} {v!r} is not a codebook label")
            if f.note and v == "other" and not val[f.note]:
                problems.append(f"{iid}: {f.column} other needs a note in {f.note}")
            if f.labels is not None and not v and not (f.column == CONDITIONAL
                                                       and val["follows_from_ontology"] != "yes"):
                problems.append(f"{iid}: {f.column} is empty")
        if form == "C" and val["follows_from_ontology"] == "yes" and not val["witness"]:
            problems.append(f"{iid}: a yes without a quoted witness is coded unsure")
        if form == "C" and val["follows_from_ontology"] != "yes" and val[CONDITIONAL]:
            problems.append(f"{iid}: {CONDITIONAL} is coded only after a yes to follows_from_ontology")
    if missing := sorted(set(want) - seen):
        problems.append(f"{len(missing)} item(s) of the package have no row (first: {missing[0]})")
    return problems


def load_adjudicated(path: Path, want: Mapping[str, str]) -> tuple[dict[str, dict[str, str]], str]:
    data = _read(path, "the adjudicated codes")
    columns, rows = _csv_rows(data, "the adjudicated codes")
    if columns != ap.SHEET_COLUMNS:
        raise AnalysisError("the adjudicated codes' columns are not the sheet's (audit_analysis, 'The adjudicated "
                            "codes')")
    problems = check_codes(rows, want)
    if problems:
        raise AnalysisError(f"the adjudicated codes have {len(problems)} problem(s), first: {'; '.join(problems[:5])}")
    return _codes(rows), _sha256(data)


def load_flags(path: Path) -> tuple[dict[str, tuple[str, str]], str]:
    data = _read(path, "the flag table")
    columns, rows = _csv_rows(data, "the flag table")
    if columns != FLAG_COLUMNS:
        raise AnalysisError(f"the flag table's columns are not {','.join(FLAG_COLUMNS)}")
    out: dict[str, tuple[str, str]] = {}
    for n, r in enumerate(rows, 2):
        rid, source, rule = ((r.get(c) or "").strip() for c in FLAG_COLUMNS)
        if not (rid and source and rule):
            raise AnalysisError(f"the flag table's line {n} has an empty run_id, source or rule")
        if rid in out:
            raise AnalysisError(f"the flag table's line {n} repeats an earlier run_id")
        out[rid] = (source, rule)
    return out, _sha256(data)


def load_key(path: Path, manifest: Mapping[str, Any], forms: Mapping[str, str], orders: Mapping[str, Sequence[str]]
             ) -> tuple[dict[str, dict[str, Any]], str]:
    """The sealed key's items by item id, after the key matches the package and the sheets; the gate's last step."""
    raw = _read(path, "the key")
    digest = _sha256(raw)
    if digest != manifest["key_sha256"]:
        raise AnalysisError("the key is not the one the package's manifest names (key_sha256)")
    try:
        key = json.loads(raw)
    except ValueError:
        raise AnalysisError("the key is not JSON") from None
    if key.get("schema") != ap.KEY_SCHEMA:
        raise AnalysisError(f"the key is not an {ap.KEY_SCHEMA} file")
    if key.get("practice") or bool(key.get("private")) != bool(manifest.get("private")):
        raise AnalysisError("the key is a practice key, or its privacy is not the package's")
    seed = key.get("shuffle_seed")
    if not isinstance(seed, str) or _sha256(seed.encode("utf-8")) != manifest["shuffle_seed_sha256"]:
        raise AnalysisError("the key's seed is not the one the package's manifest names")
    if key.get("files_sha256") != manifest["files_sha256"]:
        raise AnalysisError("the key's file hashes are not the package's")
    items = key.get("items")
    if not isinstance(items, dict) or set(items) != set(forms):
        raise AnalysisError("the key's items are not the sheets' items")
    for iid, it in items.items():
        if (it.get("sample_item") or {}).get("kind") != forms[iid]:
            raise AnalysisError(f"{iid}: the key's kind is not the sheets' form")
        if not manifest.get("private") and it["sample_item"].get("graph") in au.PRIVATE_GRAPHS:
            raise AnalysisError(f"{iid}: a public package's key holds a private graph's item")
    run_ids = [it["sample_item"].get("run_id") for it in items.values()]
    if len(set(run_ids)) != len(run_ids) or ap.assign_ids(run_ids, seed, "T") != {
            it["sample_item"]["run_id"]: iid for iid, it in items.items()}:
        raise AnalysisError("the key's item ids are not the ones its seed gives its run ids")
    if {a: list(o) for a, o in (key.get("orders") or {}).items()} != {a: list(o) for a, o in orders.items()}:
        raise AnalysisError("the key's annotator orders are not the sheets'")
    return {iid: it["sample_item"] for iid, it in items.items()}, digest


# --- the analysis -------------------------------------------------------------------------------------------------

def _sheet_order(package: Path, annotator: str) -> tuple[list[str], dict[str, str]]:
    _, rows = _csv_rows(_read(package / "annotators" / annotator / "sheet.csv", "the package's sheet"), "the sheet")
    return [r["item_id"] for r in rows], {r["item_id"]: r["form"] for r in rows}


def agreed_overrides(forms: Mapping[str, str], c1: Mapping[str, Mapping[str, str]],
                     c2: Mapping[str, Mapping[str, str]], adj: Mapping[str, Mapping[str, str]]
                     ) -> list[dict[str, str]]:
    """The adjudication's overrides of agreed codes: a field both annotators gave (non-empty) the same code, whose
    adjudicated code differs. The conditional code (``statement_wrong_or_misleading``) follows its parent: when the
    parent code (``follows_from_ontology``) is itself an override the conditional code is a consequence of it and is
    not counted again, and an empty cell is never a code (both leave it empty after anything but a yes)."""
    out: list[dict[str, str]] = []
    for i in sorted(forms):
        fields = [f.column for f in FIELDS[forms[i]]]
        for col in fields:
            given = c1[i][col]
            if not given or given != c2[i][col] or adj[i][col] == given:
                continue
            if col == CONDITIONAL and any(o["item_id"] == i for o in out):
                continue
            out.append({"item_id": i, "field": col, "annotators": given, "adjudicated": adj[i][col]})
    return out


def analyse(package: Path, sheets: Mapping[str, Path], adjudicated: Path, key: Path, flags: Path | None = None
            ) -> tuple[dict[str, Any], str]:
    """The result document and its text, after the gate (module docstring); ``key`` is opened last."""
    manifest = load_package(package)
    names = list(manifest["annotators"])
    if sorted(sheets) != sorted(names):
        raise AnalysisError(f"--sheet must name exactly the package's annotators: {', '.join(names)}")
    if len({p.resolve() for p in sheets.values()}) != 2:
        raise AnalysisError("the two annotators' sheets are one file")
    locked = {a: locked_sheet(sheets[a], package, a) for a in names}
    orders, form_maps = {}, {}
    for a in names:
        orders[a], form_maps[a] = _sheet_order(package, a)
    if form_maps[names[0]] != form_maps[names[1]]:
        raise AnalysisError("the package's two sheets list different items or forms")
    forms = form_maps[names[0]]
    by_form = dict(sorted(Counter(forms.values()).items()))
    if by_form != manifest["items_by_form"] or len(forms) != manifest["items"]:
        raise AnalysisError("the sheets' items are not the manifest's")
    c1, c2 = locked[names[0]][0], locked[names[1]][0]
    adj, adj_sha = load_adjudicated(adjudicated, forms)
    flag_map, flag_sha = load_flags(flags) if flags else (None, None)
    # every check that needs no key has passed: the key is opened now, and not before
    meta, key_sha = load_key(key, manifest, forms, orders)
    if flag_map is not None:
        b_runs = {m["run_id"] for i, m in meta.items() if forms[i] == "B"}
        if set(flag_map) != b_runs:
            raise AnalysisError(f"the flag table must name every form B item's run id once and no other "
                                f"({len(flag_map)} given, {len(b_runs)} items)")
    overrides = agreed_overrides(forms, c1, c2, adj)
    rows = [{"item_id": i, "run_id": meta[i]["run_id"], "form": forms[i], "graph": meta[i]["graph"],
             "model_group": meta[i]["model_group"], **{k: v for k, v in adj[i].items() if k != "form"}}
            for i in sorted(forms)]
    result = {
        "schema": RESULT_SCHEMA, "protocol": ap.PROTOCOL, "private": bool(manifest.get("private")),
        "tool": {"audit_analysis_sha256": _sha256(Path(__file__).read_bytes()),
                 "audit_package_sha256": _sha256(Path(ap.__file__).read_bytes())},
        "inputs": {"package_manifest_sha256": _sha256(_read(package / "manifest.json", "the manifest")),
                   "sheets_sha256": {a: locked[a][1] for a in names}, "adjudicated_sha256": adj_sha,
                   "key_sha256": key_sha, "flags_sha256": flag_sha},
        "items": {"total": len(forms), "by_form": by_form},
        "shares_note": "shares are over the sampled items, unweighted by cell (equal quotas per stratum); ci95 is the "
                       "exact Clopper-Pearson interval over items",
        "agreement": agreement(forms, c1, c2),
        "adjudication": {"overrides_of_agreed_codes": len(overrides), "overrides": overrides},
        **tables(rows, flag_map)}
    return result, report_text(result)


# --- the report ---------------------------------------------------------------------------------------------------

def _pct(c: Mapping[str, Any]) -> str:
    if c["share"] is None:
        return "-"
    return f"{100 * c['share']:.1f}% [{100 * c['ci95'][0]:.0f}-{100 * c['ci95'][1]:.0f}]"


def _table(title: str, t: Mapping[str, Any]) -> list[str]:
    lines = [f"  {title} (n = {t['n']})"]
    for lab, c in t["labels"].items():
        lines.append(f"    {lab:<26}{c['count']:>4}  {_pct(c)}")
    return lines


def report_text(r: Mapping[str, Any]) -> str:
    out = [f"Audit analysis ({r['schema']}; {'PRIVATE package' if r['private'] else 'public package'})",
           f"items: {r['items']['total']} ({', '.join(f'{k} {v}' for k, v in r['items']['by_form'].items())}); "
           f"tool sha256 {r['tool']['audit_analysis_sha256'][:16]}", r["shares_note"], "",
           "Cohen's kappa (annotator 1 against 2; * = the form's primary code; per-label kappas in the JSON)"]
    for kind, form in r["agreement"].items():
        for col, f in form["fields"].items():
            k = "undefined" if f["kappa"] is None else f"{f['kappa']:.3f}"
            out.append(f"  {kind} {col + (' *' if f['primary'] else ''):<34}n={f['n']:<4} p_o={f['observed_agreement']}"
                       f"  p_e={f['expected_agreement']}  kappa={k}")
    out += ["", f"Adjudication: {r['adjudication']['overrides_of_agreed_codes']} override(s) of a code both annotators "
            "agreed on", "", "H4 residue (form A), adjudicated codes"]
    h4 = r["H4_residue"]
    out += _table("error type, all items", h4["error_type"]["overall"]["all_items"])
    out += _table("error type, misses only (not " + "/".join(NOT_MISSES) + ")",
                  h4["error_type"]["overall"]["misses_only"])
    out += _table("structurally detectable, all items", h4["structurally_detectable"]["overall"]["all_items"])
    out += ["", "False alarms (form B), adjudicated causes"]
    fa = r["false_alarms"]
    for src, v in fa["by_source_kind"].items():
        out += _table(f"{src}: cause, all items", v["cause"]["overall"]["all_items"])
        out += _table(f"{src}: cause, answer really correct", v["cause"]["overall"]["answer_really_correct_yes"])
    for src, rules in (fa["by_rule"] or {}).items():
        for rule, v in rules.items():
            out += _table(f"{src}, rule {rule}: cause, all items", v["all_items"])
    if fa["by_rule"] is None:
        out.append("  (no flag table: no by-rule table)")
    out += ["", "H6 shared premise (form C)"]
    h6 = r["H6_shared_premise"]
    out += _table("follows from the ontology", h6["follows_from_ontology"]["overall"]["all_items"])
    out += _table("statement wrong or misleading, after a yes",
                  h6["statement_wrong_or_misleading"]["overall"]["follows_yes"])
    return "\n".join(out) + "\n"


# --- output -------------------------------------------------------------------------------------------------------

def check_out(out: Path) -> None:
    """The refusals that need no package: run before anything is read."""
    if out.resolve().is_relative_to(ROOT):
        raise AnalysisError("--out must be outside the repository: analysis outputs are the owner's")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise AnalysisError("--out exists and is not an empty directory")


def write_outputs(out: Path, files: Mapping[str, str], private: bool) -> None:
    """Stage the files beside ``out`` (directory 700, files 600) and move them into place; a failure leaves nothing."""
    check_out(out)
    if private and out.parent.exists() and not _closed(out.parent):
        raise AnalysisError("a private package's analysis goes under a directory open to its owner only")
    out.parent.mkdir(parents=True, exist_ok=True)
    if private and not _closed(out.parent):
        raise AnalysisError("a private package's analysis goes under a directory open to its owner only")
    staging = Path(tempfile.mkdtemp(prefix=".audit-analysis-", dir=out.parent))
    try:
        os.chmod(staging, 0o700)
        for name, text in files.items():
            fd = os.open(staging / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
        os.rename(staging, out)  # onto a missing or empty directory
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    ap_ = argparse.ArgumentParser(description="Analyse the audit: kappa and the adjudicated codes' counts and shares.")
    ap_.add_argument("--package", type=Path, required=True,
                     help="the audit package directory (manifest.json, annotators/)")
    ap_.add_argument("--sheet", action="append", default=[], metavar="NAME=FILE",
                     help="an annotator's filled sheet; give both annotators' (repeatable)")
    ap_.add_argument("--adjudicated", type=Path, required=True, help="the adjudicated codes (CSV, the sheet's columns)")
    ap_.add_argument("--key", type=Path, required=True, help="the sealed key; opened only after both sheets lock")
    ap_.add_argument("--out", type=Path, required=True, help="a new or empty directory (mode 700; files 600)")
    ap_.add_argument("--flags", type=Path, help="optional flag table (CSV run_id,source,rule) for the by-rule table")
    ap_.add_argument("--print", action="store_true", help="print the text table (a public package's only)")
    args = ap_.parse_args(argv)
    sheets: dict[str, Path] = {}
    for spec in args.sheet:
        name, eq, path = spec.partition("=")
        if not (eq and name and path) or name in sheets:
            ap_.error("--sheet takes NAME=FILE, once per annotator")
        sheets[name] = Path(path)
    try:
        check_out(args.out)
        result, text = analyse(args.package, sheets, args.adjudicated, args.key, args.flags)
        private = result["private"]
        write_outputs(args.out, {"audit-analysis.json": json.dumps(result, indent=1, sort_keys=True) + "\n",
                                 "audit-analysis.txt": text}, private)
    except (AnalysisError, OSError) as e:
        print(f"audit analysis refused: {e}", file=sys.stderr)
        return 2
    if private:
        print("wrote the private package's analysis (mode 600; nothing of it printed)")
    else:
        print(f"wrote audit-analysis.json and audit-analysis.txt to {args.out}")
        if args.print:
            print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
