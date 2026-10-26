#!/usr/bin/env python3
"""Did a change of copy labels (and nothing else) separate two e189 looks?

    python3 -I labels_only_diff.py OLD NEW [--endpoint-copies PATH]

OLD and NEW are two e189 ``results.json`` files of the same look, made before and after the endpoint copies
(endpoint-copies.json) gave the single-endpoint OUTs' copies their replica names and classes. Prints ONLY

    identical-outside-labels: true
or
    identical-outside-labels: false
    ["stores", "copies", "rhea-c06", "outs"]      (one line per differing JSON key path, sorted; a JSON array of keys)

Never a value and never a count: only key names (an arm or OUT or copy label is a key; no number or text that is a value),
a list element is the key ``[]`` in a path, two lists of different length differ at the list's own path, and the messages
on stderr are fixed sentences that echo no input (also for a bad command line). Exit 0 whenever it ran (whatever the
answer), 2 on bad input (an unreadable file, a file that is not an e189 results object, an invalid endpoint-copies file,
two table endpoints sharing a 12-hex label, a bad command line).

How OLD is made comparable (the endpoint-copies table is read from ../endpoint-copies.json, or --endpoint-copies; its
entries give the mapping of a label "endpoint sha256:<first 12 hex of endpoint_sha256>" to its replica). Everything else
is compared exactly (JSON equality, key by key), so anything not listed here that differs is printed.

Mapped in OLD (a label of the four table copies becomes its replica), only where check_stores puts a copy label and a
table-named one can occur (an OUT with store changes needs proofs for every endpoint, so its labels, the
``store_changes`` and ``branch_phase_copies``, are never table-named):
  * the keys of ``stores.copies``: a mapped key that meets an existing key (the proof-named entry of the same replica)
    merges the way check_stores merges one label's OUTs: ``endpoints`` and ``outs`` the sorted union, ``listed_by`` the
    union (an OUT in both is a difference); the mapped entry must carry no ``proofs``, ``periods``, ``split``,
    ``switch_instants`` or ``not_checked`` (the table is no proof), or it is a difference;
  * the keys of ``rows`` in ``stores.shares``, ``shares_by_out``, ``attempt_shares`` and ``attempt_shares_by_out``: a
    collision merges ``n`` by sum with the same ``of`` and the share recomputed as n/of, but only when both components are
    well-formed rows (exactly n, of, share; n and of integers; share exactly n/of), else a difference;
Not mapped: ``stores.copies.*.endpoints`` (they stay endpoint labels in both), the copy objects of
``stores.branch_phase_copies`` and any other string (an OUT with a branch phase has store changes, so every endpoint of
its chain is proof-named: a table label there is a difference).

Excluded from the comparison, each because it carries a copy class label or the provenance of the class sources:
  * ``stores.config.copy_classes`` and ``stores.config.endpoint_copies``: the path and sha256 of the copy-classes file
    (rebuilt for the table's copies) and of the endpoint-copies file (new); and ``stores.config.private_copy_classes``
    and ``stores.config.private_endpoint_copies``, the basename and sha256 of a privately pinned graph's own class
    table and endpoint-to-copy table (each present only in a call that gave one);
  * ``stores.class_shares`` and ``stores.class_shares_by_out``, the ``rows`` of each share (their keys are class labels:
    a row 'no copy class (...)' becomes the replica's class, and rows of one class merge, so a row's own n and share may
    change). What is still compared there: each share's ``attempts`` and ``without record``, the sum of its rows' ``n``,
    the distinct denominators ``of`` of its rows, and whether every row is a well-formed share row.
NOT excluded, although they carry class labels: the ``class``, ``label``, ``basis`` and ``source`` of the copy objects in
``stores.branch_phase_copies``. Those are proof-named copies, whose entries the rebuilt copy-classes.json leaves byte for
byte as they were, so OLD and NEW agree there; a difference is printed (a rebuild that changed an existing entry is
exactly what this tool should show).
Code-version fields (the lead's list names the e189 commit, tree and script hashes): none exist in the results.json
that exploratory.py writes (its ``exploratory.code`` is the frozen code root's identity, which a label change leaves
alone, and the e189 commit, tree and script hashes are not in the file), so none is excluded; a field a driver adds to the
file beyond that shows as a differing path.

The tool reads two result files the operator names and prints key paths; run it only where the operator's rules allow
reading results (it is for the operator's private directory). Its key paths hold copy labels: on the results of a call
that used a privately pinned graph's own tables (``stores.config.private_copy_classes`` or ``private_endpoint_copies``
in either input) they are private copy names, and through ``shares_by_out`` pairs of an OUT and a private copy, so its
"never a value" output is then not for relaying; it prints one more line, PRIVATE_NOTE, saying so. The tests use small
synthetic fixtures."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
LABEL_PREFIX = "endpoint sha256:"
PROVENANCE = (("stores", "config", "copy_classes"), ("stores", "config", "endpoint_copies"),
              ("stores", "config", "private_copy_classes"), ("stores", "config", "private_endpoint_copies"))
ROW_SECTIONS = ("shares", "shares_by_out", "attempt_shares", "attempt_shares_by_out")
CLASS_SECTIONS = ("class_shares", "class_shares_by_out")
MERGE_ONLY = {"endpoints", "outs", "listed_by", "proofs"}  # the keys a table-named copies entry may carry


PRIVATE_RECORDS = ("private_copy_classes", "private_endpoint_copies")
PRIVATE_NOTE = ("private inputs: these results name private copies, so the key paths above can name them; this output "
                "is as private as the results")


def _has_private_record(doc: Any) -> bool:
    """Whether a results document records a privately pinned graph's own class table or endpoint-to-copy table."""
    config = ((doc.get("stores") or {}).get("config") or {}) if isinstance(doc, dict) else {}
    return isinstance(config, dict) and any(name in config for name in PRIVATE_RECORDS)


class BadInput(Exception):
    """The inputs cannot be compared (exit 2). The message names no value."""


def _load_copy_classes() -> Any:
    spec = importlib.util.spec_from_file_location("e189_copy_classes_for_diff", HERE.parent / "copy_classes.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def label_mapping(table_path: Path) -> dict[str, str]:
    """{"endpoint sha256:<12 hex>": replica} from the endpoint-copies file, validated by copy_classes."""
    module = _load_copy_classes()
    try:
        entries, _ = module.read_endpoint_copies(table_path)
    except (module.Refused, OSError):
        raise BadInput("the endpoint copies cannot be read or are invalid") from None
    mapping: dict[str, str] = {}
    for e in entries:
        label = LABEL_PREFIX + e["endpoint_sha256"][:12]
        if label in mapping:
            raise BadInput("two endpoint copies share their 12-hex label")
        mapping[label] = e["replica"]
    return mapping


def _read(path: Path) -> dict:
    try:
        doc = json.loads(Path(path).read_bytes())
    except (OSError, ValueError):
        raise BadInput("a results file cannot be read as JSON") from None
    if not isinstance(doc, dict) or not isinstance(doc.get("stores"), dict):
        raise BadInput("a results file is not an e189 results object with a stores section")
    return doc


def _row_ok(row: Any) -> bool:
    """A share row as share_of writes it: exactly n, of and share, n and of integers (not booleans), 1 <= n <= of (share_of
    writes a row only for a label with an attempt), and share exactly n / of."""
    return (isinstance(row, dict) and set(row) == {"n", "of", "share"}
            and all(type(row[k]) is int for k in ("n", "of")) and 1 <= row["n"] <= row["of"]
            and type(row["share"]) is float and row["share"] == row["n"] / row["of"])


def _merge_rows(target: dict, extra: dict, forced: set, path: tuple) -> None:
    """Add the row ``extra`` ({n, of, share}) to ``target`` the way one label's rows add: n by sum, the same ``of``. A
    component that is not a well-formed row (its share is not n / of, say) is never merged: the recomputed share would
    hide it."""
    if not _row_ok(target) or not _row_ok(extra) or target["of"] != extra["of"]:
        forced.add(path)
        return
    target["n"] += extra["n"]
    target["share"] = target["n"] / target["of"]


def _relabel_rows(rows: Any, mapping: dict[str, str], forced: set, path: tuple) -> Any:
    if not isinstance(rows, dict):
        return rows
    out: dict[str, Any] = {k: v for k, v in rows.items() if k not in mapping}
    for key, value in rows.items():
        if key in mapping:
            target = mapping[key]
            if target in out:
                if isinstance(out[target], dict) and isinstance(value, dict):
                    out[target] = copy.deepcopy(out[target])
                    _merge_rows(out[target], value, forced, (*path, target))
                else:
                    forced.add((*path, target))
            else:
                out[target] = value
    return out


def _relabel_share(share: Any, mapping: dict[str, str], forced: set, path: tuple) -> Any:
    if isinstance(share, dict) and isinstance(share.get("rows"), dict):
        share = {**share, "rows": _relabel_rows(share["rows"], mapping, forced, (*path, "rows"))}
    return share


def _relabel_copies(copies: Any, mapping: dict[str, str], forced: set) -> Any:
    if not isinstance(copies, dict):
        return copies
    path = ("stores", "copies")
    out: dict[str, Any] = {k: copy.deepcopy(v) for k, v in copies.items() if k not in mapping}
    for key, value in copies.items():
        if key not in mapping:
            continue
        replica = mapping[key]
        if not isinstance(value, dict) or set(value) - MERGE_ONLY or value.get("proofs") != []:
            forced.add((*path, replica))  # a table-named entry carries no proof: anything else is not the table's
            out.setdefault(replica, copy.deepcopy(value))
            continue
        if replica not in out:
            out[replica] = copy.deepcopy(value)
            continue
        have = out[replica]
        if not isinstance(have, dict):
            forced.add((*path, replica))
            continue
        for field in ("endpoints", "outs"):
            if isinstance(have.get(field), list) and isinstance(value.get(field), list):
                have[field] = sorted({*have[field], *value[field]})
            else:
                forced.add((*path, replica, field))
        mine, theirs = have.get("listed_by"), value.get("listed_by")
        if isinstance(mine, dict) and isinstance(theirs, dict) and not set(mine) & set(theirs):
            have["listed_by"] = {**mine, **theirs}
        else:
            forced.add((*path, replica, "listed_by"))
    return out


def normalise(doc: dict, mapping: dict[str, str] | None) -> tuple[dict, set]:
    """The document without the excluded fields; with ``mapping`` (OLD only) its table-named labels as replicas. Also
    the key paths that cannot be reconciled (a merge that does not fit the rules)."""
    doc = copy.deepcopy(doc)
    forced: set[tuple] = set()
    stores = doc["stores"]
    for *parents, last in PROVENANCE:
        node: Any = doc
        for p in parents:
            node = node.get(p) if isinstance(node, dict) else None
        if isinstance(node, dict):
            node.pop(last, None)
    for name in CLASS_SECTIONS:  # rows keyed by class labels: compare what does not depend on them
        section = stores.get(name)
        if isinstance(section, dict):
            for key in list(section):
                if name == "class_shares_by_out":
                    inner = section[key]
                    for arm in (list(inner) if isinstance(inner, dict) else ()):
                        inner[arm] = _class_totals(inner[arm])
                else:
                    section[key] = _class_totals(section[key])
    if mapping is not None:
        if "copies" in stores:
            stores["copies"] = _relabel_copies(stores["copies"], mapping, forced)
        for name in ROW_SECTIONS:
            section = stores.get(name)
            if not isinstance(section, dict):
                continue
            for key, value in list(section.items()):
                if name.endswith("_by_out") and isinstance(value, dict):
                    for arm, share in list(value.items()):
                        value[arm] = _relabel_share(share, mapping, forced, ("stores", name, key, arm))
                else:
                    section[key] = _relabel_share(value, mapping, forced, ("stores", name, key))
    return doc, forced


def _class_totals(share: Any) -> Any:
    """A class share without its class-labelled rows: what stays is the attempts, those without a record, the rows' total
    (n summed), the denominators the rows use (the distinct ``of``) and whether every row is a well-formed share row;
    none of these depends on which class label a copy has (rows of one class merge, so a row's own n and share may)."""
    if not isinstance(share, dict) or not isinstance(share.get("rows"), dict):
        return share
    rows = list(share["rows"].values())
    ok = all(_row_ok(r) for r in rows)
    return {**{k: v for k, v in share.items() if k != "rows"},
            "rows_total_n": sum(r["n"] for r in rows) if ok else None,
            "rows_of": sorted({r["of"] for r in rows}) if ok else None, "rows_well_formed": ok}


def differing_paths(a: Any, b: Any, path: tuple = ()) -> set[tuple]:
    """The key paths where a and b differ (a list element is the key '[]'; lists of different lengths differ at the
    list itself)."""
    if isinstance(a, dict) and isinstance(b, dict):
        found: set[tuple] = set()
        for key in set(a) | set(b):
            if key not in a or key not in b:
                found.add((*path, key))
            else:
                found |= differing_paths(a[key], b[key], (*path, key))
        return found
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return {path}
        found = set()
        for x, y in zip(a, b, strict=True):
            found |= differing_paths(x, y, (*path, "[]"))
        return found
    same = type(a) is type(b) and a == b
    return set() if same else {path}


def compare(old: dict, new: dict, mapping: dict[str, str]) -> list[tuple]:
    """The sorted key paths where OLD (its table-named labels mapped) and NEW differ outside the excluded fields."""
    old_n, forced = normalise(old, mapping)
    new_n, _ = normalise(new, None)
    return sorted(differing_paths(old_n, new_n) | forced, key=lambda p: json.dumps(p))


class _Parser(argparse.ArgumentParser):
    """argparse echoes an unrecognised argument in its error: here the message is a fixed sentence."""

    def error(self, message: str) -> None:  # noqa: ARG002
        print("labels_only_diff: bad command line (OLD NEW [--endpoint-copies PATH])", file=sys.stderr)
        raise SystemExit(2)


def main(argv: list[str] | None = None) -> int:
    parser = _Parser(description=__doc__.split("\n\n")[0])
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--endpoint-copies", type=Path, default=HERE.parent / "endpoint-copies.json",
                        help="the endpoint-copies file (default: beside the exploratory script)")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 2 if exc.code else 0
    try:
        mapping = label_mapping(args.endpoint_copies)
        old, new = _read(args.old), _read(args.new)
        paths = compare(old, new, mapping)
    except BadInput as exc:
        print(f"labels_only_diff: bad input: {exc}", file=sys.stderr)
        return 2
    print(f"identical-outside-labels: {'true' if not paths else 'false'}")
    for path in paths:
        print(json.dumps(list(path)))
    if any(_has_private_record(doc) for doc in (old, new)):  # key paths hold labels: here, a private graph's copy names
        print(PRIVATE_NOTE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
