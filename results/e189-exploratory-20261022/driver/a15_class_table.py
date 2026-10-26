#!/usr/bin/env python3
"""A names-only table of the store copies behind each private-table OUT's outputs and behind its replay, with their speed classes
(science's ruling, 26 Oct 2026: it replaces the cross-tab, which has no copy-class fields).

    python3 -I a15_class_table.py --rows ROWS.tsv --classes CLASSES --classes-sha256 HEX --endpoints ENDPOINTS --endpoints-sha256 HEX \\
        --description-pins PINS --description-pins-sha256 HEX --out TABLE.tsv

Prints nothing but ``ok rows=N`` (exit 0) or one line ``REFUSED <code>`` on stderr (exit 2); no name, hash or value is echoed.

ROWS.tsv: one line per OUT-phase, ``out<TAB>phase<TAB>manifest.json path<TAB>run.json path`` (absolute paths without ``..``; the
manifest is the OUT's or its view's, run.json the replay results folder's). The pair (out, phase) must be one of the pairs
a15_replay_matrices.py serves. The three tables are read only after their sha256 matches the pin given: CLASSES the private copy
classes (copy_classes.py's document: replica -> class), ENDPOINTS the private endpoint copies (graph, endpoint sha256 -> replica;
the same two files the analysis gets with --private-copy-classes and --private-endpoint-copies), PINS ``copy<TAB>run-description
sha256`` lines (``#`` lines allowed): the replay launchers' copy table, the fallback for the replay copy.

OUTPUT COPIES come from the manifest alone: ``fingerprint.store.name`` and ``fingerprint.store.endpoint``, and ``store_changes`` (each
change's ``from`` and ``to`` side name one endpoint, in ``endpoint`` or ``store.endpoint``; the sides chain, and the manifest's endpoint
is the chain's first or last). A copy is the replica the endpoint table gives for (graph, sha256 of the endpoint string). Where the
manifest gives it, the attempts made on a copy are the difference of the records made before the next change and before this one
(``pre_change_records``, which must be the length of ``pre_change_run_ids``); the chain's last copy has none (``-``).

THE REPLAY COPY: run.json is read for ``store.version`` and ``store.description_sha256`` and for nothing else (gated_replay.py's
write_run records ``store.version`` as the ``version`` of the store description the replay read; the description's sha256 names no copy
by itself). The copy is ``store.version`` only when that is one of the replica names of CLASSES or ENDPOINTS (``replay_source``
``version``); else the copy PINS gives for the description sha256 (``description-pin``); else ``unknown``. A version and a pin that
name different copies refuse (``replay-copy-conflict``).

CLASSES. A copy's class is its entry in CLASSES (fast, medium, slow); a copy that has none, or a null class, is ``unknown``.
``same_class`` is ``unknown`` when the replay's class or any output copy's is unknown; else ``yes`` when every output copy's class is the
replay's, ``no`` when none is, ``mixed`` otherwise.

The TABLE (mode 600, must not exist): a header line and ``out phase output_copies output_attempts output_classes replay_copy
replay_class same_class`` per row (the first three comma lists of the chain, in order; ``unknown`` for a copy no table names).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIMIT = 1 << 24
SHA = re.compile(r"[0-9a-f]{64}")
COPY_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
UNKNOWN = "unknown"
HEADER = "out\tphase\toutput_copies\toutput_attempts\toutput_classes\treplay_copy\treplay_source\treplay_class\tsame_class\n"


def _load(name: str, path: Path):
    """A module of ``path`` loaded by path (``-I`` leaves this directory off sys.path); one already loaded from that very file is reused."""
    have = sys.modules.get(name)
    if have is not None and getattr(have, "__file__", None) and Path(have.__file__).resolve() == path.resolve():
        return have
    if have is not None:
        name = f"_class_table_{name}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


a15 = _load("a15_replay_matrices", HERE / "a15_replay_matrices.py")
rm = a15.rm
cc = _load("e189_copy_classes", HERE.parent / "copy_classes.py")
Refused = rm.Refused


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_pinned(path: str, pin: str, what: str) -> bytes:
    """The bytes of the file when their sha256 is the pin (a regular file, not a link); else Refused."""
    if not SHA.fullmatch(pin):
        raise Refused("pin-malformed", what=what)
    data = rm.read_small(Path(path), LIMIT)
    if hashlib.sha256(data).hexdigest() != pin:
        raise Refused("pin-differs", what=what)
    return data


def absolute_clean(path: str, what: str) -> str:
    if not path.startswith("/") or ".." in path.split("/") or any(c in path for c in "\n\r\t\0"):
        raise Refused("path-not-clean", what=what)
    return path


def read_pins(raw: bytes) -> dict[str, str]:
    """{run-description sha256: copy key} from ``copy<TAB>sha256`` lines; a repeated copy or sha256 refuses."""
    by_sha: dict[str, str] = {}
    copies: set[str] = set()
    for ln in raw.decode("utf-8").splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        f = ln.split("\t")
        if len(f) != 2 or not COPY_NAME.fullmatch(f[0]) or not SHA.fullmatch(f[1]) or f[0] in copies or f[1] in by_sha:
            raise Refused("pins-malformed")
        copies.add(f[0])
        by_sha[f[1]] = f[0]
    if not by_sha:
        raise Refused("pins-malformed")
    return by_sha


def side_endpoint(side) -> str:
    """One store-change side's endpoint (``endpoint`` and ``store.endpoint`` must name one)."""
    if not isinstance(side, dict) or side.get("store_client"):
        raise Refused("store-change-side")
    store = side.get("store") if isinstance(side.get("store"), dict) else {}
    found = {e for e in (side.get("endpoint"), store.get("endpoint")) if isinstance(e, str) and e}
    if len(found) != 1:
        raise Refused("store-change-side")
    return found.pop()


def manifest_chain(manifest) -> tuple[str, list[str], list[str]]:
    """(graph, the chain of endpoints in order of use, the attempts made on each: a count or ``-``) from a manifest."""
    fp = manifest.get("fingerprint") if isinstance(manifest, dict) else None
    store = fp.get("store") if isinstance(fp, dict) else None
    graph = store.get("name") if isinstance(store, dict) else None
    endpoint = store.get("endpoint") if isinstance(store, dict) else None
    if not (isinstance(graph, str) and graph and isinstance(endpoint, str) and endpoint):
        raise Refused("manifest-store")
    raw = manifest.get("store_changes")
    if raw is None or raw == []:
        return graph, [endpoint], ["-"]
    if not isinstance(raw, list):
        raise Refused("manifest-store-changes")
    chain: list[str] = []
    before: list[int] = []
    for ch in raw:
        if not isinstance(ch, dict):
            raise Refused("manifest-store-changes")
        a, b = side_endpoint(ch.get("from")), side_endpoint(ch.get("to"))
        ids = ch.get("pre_change_run_ids")
        n = ch.get("pre_change_records")
        if (a == b or not isinstance(ids, list) or type(n) is not int or n != len(ids) or (chain and chain[-1] != a)
                or (before and n < before[-1])):
            raise Refused("manifest-store-changes")
        if not chain:
            chain.append(a)
        chain.append(b)
        before.append(n)
    if endpoint not in (chain[0], chain[-1]):
        raise Refused("manifest-endpoint-not-in-chain")
    cuts = [0, *before]
    return graph, chain, [str(cuts[i + 1] - cuts[i]) for i in range(len(before))] + ["-"]


def class_of(copy: str, classes: dict) -> str:
    entry = classes.get(copy)
    speed = entry.get("class") if isinstance(entry, dict) else None
    return speed if speed in cc.CLASSES else UNKNOWN


def replay_copy_of(run, classes: dict, replicas: dict, pins: dict) -> tuple[str, str]:
    """(copy, source) of a replay from its run.json: ``store.version`` when it is a replica name of the tables, else the pin of the
    description sha256, else unknown; a version and a pin that disagree refuse."""
    store = run.get("store") if isinstance(run, dict) else None
    if not isinstance(store, dict):
        raise Refused("run-json-description")
    version, description = store.get("version"), store.get("description_sha256")
    if not (isinstance(description, str) and SHA.fullmatch(description)):
        raise Refused("run-json-description")
    names = set(classes) | set(replicas.values())
    named = version if isinstance(version, str) and COPY_NAME.fullmatch(version) and version in names else None
    pinned = pins.get(description)
    if named is not None and pinned is not None and named != pinned:
        raise Refused("replay-copy-conflict")
    if named is not None:
        return named, "version"
    if pinned is not None:
        return pinned, "description-pin"
    return UNKNOWN, UNKNOWN


def build_row(out: str, phase: str, manifest_path: str, run_path: str, classes: dict, replicas: dict, pins: dict) -> str:
    manifest = json.loads(rm.read_small(Path(manifest_path), LIMIT).decode("utf-8"))
    graph, chain, attempts = manifest_chain(manifest)
    copies = [replicas.get((graph, sha256_text(ep)), UNKNOWN) for ep in chain]
    speeds = [class_of(c, classes) if c != UNKNOWN else UNKNOWN for c in copies]
    replay, source = replay_copy_of(json.loads(rm.read_small(Path(run_path), LIMIT).decode("utf-8")), classes, replicas, pins)
    replay_class = class_of(replay, classes) if replay != UNKNOWN else UNKNOWN
    if UNKNOWN in speeds or replay_class == UNKNOWN:
        same = UNKNOWN
    elif all(s == replay_class for s in speeds):
        same = "yes"
    elif any(s == replay_class for s in speeds):
        same = "mixed"
    else:
        same = "no"
    return "\t".join([out, phase, ",".join(copies), ",".join(attempts), ",".join(speeds), replay, source, replay_class, same]) + "\n"


def run(args: argparse.Namespace) -> int:
    pairs = a15.out_phases()
    classes = cc.loads(read_pinned(args.classes, args.classes_sha256, "classes"))["copies"]
    entries = cc.loads_endpoint_copies(read_pinned(args.endpoints, args.endpoints_sha256, "endpoints"))
    replicas = {(e["graph"], e["endpoint_sha256"]): e["replica"] for e in entries}
    pins = read_pins(read_pinned(args.description_pins, args.description_pins_sha256, "description-pins"))
    out_path = Path(os.path.abspath(args.out))
    if os.path.lexists(out_path):
        raise Refused("out-exists")
    rows, seen = [], set()
    for ln in rm.read_small(Path(args.rows), LIMIT).decode("utf-8").splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        f = ln.split("\t")
        if len(f) != 4 or (f[0], f[1]) not in pairs or (f[0], f[1]) in seen:
            raise Refused("rows-malformed")
        seen.add((f[0], f[1]))
        rows.append(build_row(f[0], f[1], absolute_clean(f[2], "manifest"), absolute_clean(f[3], "run.json"), classes, replicas, pins))
    if not rows:
        raise Refused("rows-malformed")
    fd = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(HEADER + "".join(rows))
    print(f"ok rows={len(rows)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = rm.QuietParser(description=__doc__.split("\n\n")[0])
    for name in ("rows", "classes", "classes-sha256", "endpoints", "endpoints-sha256", "description-pins", "description-pins-sha256", "out"):
        ap.add_argument(f"--{name}", required=True)
    args = ap.parse_args(argv)
    previous = os.umask(0o077)
    try:
        try:
            return run(args)
        except Refused as exc:
            what = f" what={exc.counts['what']}" if "what" in exc.counts else ""
            print(f"REFUSED {exc.code}{what}", file=sys.stderr)
            return 2
        except cc.Refused as exc:  # a malformed private table: the code only (its text could name a private copy)
            print("REFUSED table-malformed", file=sys.stderr)
            return 2
        except (OSError, EOFError, ValueError, KeyError, TypeError, AttributeError, zlib.error) as exc:
            print(f"REFUSED io-error class={type(exc).__name__}", file=sys.stderr)
            return 2
    finally:
        os.umask(previous)


if __name__ == "__main__":
    sys.exit(main())
