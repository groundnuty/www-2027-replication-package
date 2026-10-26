#!/usr/bin/env python3
"""The freeze-binding check (E8.1, design log): every hash, commit and [FILL] the pre-registration cites,
checked against the live artefact.

    uv run python scripts/freeze_binding_check.py [--science-repo DIR] [--commit REF] [--registry FILE]
        [--repo NAME=DIR ...] [--md OUT.md] [--json OUT.json] [--allow-fill] [--python EXE]

The documents (``paper/preregistration.md`` and the change log beside it) are read from the science
repo at ``--commit`` with ``git show``, never from a working tree. Every sha256 (64 hex), short hash,
git commit and [FILL] placeholder they cite is extracted with the text around it.

The registry (``data/freeze/registry.json``, reviewed by eye) maps the *phrase* that names an
artefact to the artefact: a file at a git ref, a function of the code at a ref (the generator's own
gold-row hash, the code digest, the feedback-format hash, the calibration fingerprint), or a commit
that must be reachable from a ref. It never holds a hash value, and anchors never name one, so a stale
value in the document is a mismatch, not a missing entry. The check recomputes each value from the
artefact itself and only then compares it with the cited one; a value written anywhere else (in the
document, a manifest, a commit message, a content-addressed file name) is never the expected value.

The one exception is a value that exists nowhere but in a build record: the QLever image digest and
git commit are not artefacts in git, only fields of each store's committed ``manifest.json``. A
``json_hash`` resolver reads such a field (``pointer``) and extracts the hash from its string with a
``regex`` whose single ``(?P<value>...)`` group names no literal digit or letter, so the pattern cannot
encode the expected value. It binds the document to the build record, not to the registry or the
binary: it proves the document quotes what the build wrote down, not that the record is right.

Statuses: ``match``; ``mismatch``; ``unresolvable`` (no registry entry, two entries, or the artefact
cannot be recomputed; with why); ``unresolved`` (a [FILL], with the artefact's current value when the
registry names it); ``ignored`` (a [FILL] the registry marks as prose about the marker); ``historical`` (a hash the
registry marks as cited for the record only, bound to nothing: a replaced value, a stale pin, a removed copy;
change-log entry 126). Exit 1 on any
mismatch or unresolvable entry, and on any unresolved one unless ``--allow-fill``; exit 2 on a usage
error (an unreadable document or registry).

Code-running resolvers extract only ``harness/ scripts/ pyproject.toml uv.lock`` at the ref (never
``heldout/`` or data) and run the function in a separate interpreter (``-I -B``) that fails if any
``harness`` module loads from outside the extracted tree.

**Facts** are cited values that are not hashes (N1 = 20,000, the seeds, the seed scheme, the number of
calibration configurations). A fact entry gives a pattern whose ``value`` group reads the cited value
(the group may name no literal digit or letter, so it never encodes the expected value), and a value
resolver that recomputes it: a module attribute or function result of the code at a ref
(``python_value``), or a field of a JSON artefact at a ref (``json_value``).

An input that is not in git (the calibration's bank items) is read with ``{"$remote": {"host", "path"}}``:
``ssh <host> tar -c <path>`` into the run's temporary directory, read-only on the host, and every file
read is listed with its sha256 in the report.

**Bindings** (at ``--bind-ref REF``, the experiments ref that runs: by default the experiments repo's HEAD, at the
freeze the freeze tag) are pins the code holds of artefacts, checked against the artefacts as the citations are
(design ruling, design log, 20 Oct): the dev
grid's ``code.check_contract_sha256`` (data/runs/dev-grid/grid.json at REF) must be the sha256 of
paper/s01-check-contract.md at the science ``--commit``; the frozen analysis config's pin
(data/freeze/frozen-config.json at REF, which a confirmatory analysis run compares with its config) must be the sha256
of the config file it names at REF; the pin's ``fault_rule`` (the test runs' fault rule, change-log entries 148 and
149) must be the sha256 of the fault-rule file it names, and that file's pins of the dev baselines and the probe sets
in git the sha256 of each. A pinned [FILL] is unresolved; anything unreadable is unresolvable. They are
checked in every run unless ``--no-bindings`` is given, which the report records ("not checked").
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DOCS = ("paper/preregistration.md", "paper/design-changelog.md")
DEFAULT_COMMIT = "refs/remotes/origin/main"
DEFAULT_REGISTRY = ROOT / "data" / "freeze" / "registry.json"
CODE_PATHS = ("harness", "scripts", "pyproject.toml", "uv.lock")
STATUSES = ("match", "mismatch", "unresolvable", "unresolved", "ignored", "historical")
TIMEOUT = 900
REMOTE_SHELL = ("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20")

HEX = re.compile(r"(?<![0-9A-Za-z_])[0-9a-f]{7,64}(?![0-9A-Za-z_])")
FILL = re.compile(r"\[FILL\b[^\]]*\]", re.IGNORECASE)  # [FILL], [FILL: x], [FILL – x], [fill]
BARE_FILL = re.compile(r"\bFILL\b")  # a backstop: an unbracketed FILL is a placeholder too
DIGEST = re.compile(r"[0-9a-f]{7,64}")
HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*$")
DELIMITERS = ";|"
COMMIT_WORDS = re.compile(r"\b(?:commit|commits|committed|merged?|revision|HEAD|branch|tag)\b", re.IGNORECASE)
HASH_WORDS = re.compile(r"\b(?:sha-?256|sha|hash(?:es)?|digest|fingerprint|checksum|file|manifest|render|bundle|"
                        r"request)\b", re.IGNORECASE)
# A token with only digits or only letters counts as a citation only in explicit context (the seed 20261026 does not).
RIGHT_AFTER_KEYWORD = re.compile(r"(?:sha-?256|hash|digest|commit|fingerprint|checksum)\W{0,3}$", re.IGNORECASE)


class RegistryError(Exception):
    pass


class ResolveError(Exception):
    pass


# -- extraction --------------------------------------------------------------------------

@dataclass(frozen=True)
class Citation:
    doc: str
    line: int
    col: int
    kind: str  # sha256 | short_hash | commit | hex | fill
    token: str
    before: str  # the clause before the token (back to ; or | or the previous citation)
    after: str  # the clause after it
    blanked: str  # the whole line, every citation replaced by <HEX> or <FILL>
    section: str
    # the sentence up to the token, from the last ". " or "| " (or the line's start), earlier citations blanked: it
    # tells apart lists on one line whose labels come before the hashes, where ``before`` stops at the previous citation
    sentence: str = ""


def _kind(token: str, before: str, after: str) -> str:
    if len(token) == 64:
        return "sha256"
    last = {name: max((m.end() for m in words.finditer(before)), default=-1)
            for name, words in (("commit", COMMIT_WORDS), ("hash", HASH_WORDS))}
    if last["commit"] > last["hash"] and len(token) <= 40:
        return "commit"
    if last["hash"] >= 0 or after.startswith(("…", "...")):
        return "short_hash"
    return "hex"


def extract(doc: str, text: str) -> tuple[list[Citation], list[dict[str, Any]]]:
    """Every cited hash, commit and [FILL] in ``text``, and every hex-like token skipped (with why)."""
    citations: list[Citation] = []
    skipped: list[dict[str, Any]] = []
    section = ""
    for number, line in enumerate(text.splitlines(), start=1):
        heading = HEADING.match(line)
        if heading:
            section = heading.group(1)
        spans: list[tuple[int, int, str]] = [(m.start(), m.end(), "fill") for m in FILL.finditer(line)]
        masked = line
        for start, end, _ in spans:  # hex inside a [FILL: ...] is part of the placeholder
            masked = masked[:start] + " " * (end - start) + masked[end:]
        spans += [(m.start(), m.end(), "fill") for m in BARE_FILL.finditer(masked)]
        for m in HEX.finditer(masked):
            token, start = m.group(), m.start()
            explicit = (line[start - 1:start] == "`" or line[m.end():].startswith(("…", "..."))
                        or bool(RIGHT_AFTER_KEYWORD.search(line[:start])))
            has_digit, has_letter = any(c.isdigit() for c in token), any(c.isalpha() for c in token)
            if len(token) == 64 or (has_digit and has_letter) or explicit:
                spans.append((start, m.end(), "hex"))
            else:
                why = "digits only" if has_digit else "letters only"
                skipped.append({"doc": doc, "line": number, "token": token,
                                "why": f"{why}, no backtick, ellipsis or hash/commit keyword"})
        spans.sort()
        blanked, shift, marks = line, 0, []
        for start, end, what in spans:
            mark = "<FILL>" if what == "fill" else "<HEX>"
            marks.append(start + shift)
            blanked = blanked[:start + shift] + mark + blanked[end + shift:]
            shift += len(mark) - (end - start)
        for i, (start, end, what) in enumerate(spans):
            left = max([line.rfind(d, 0, start) + 1 for d in DELIMITERS] + [spans[i - 1][1] if i else 0])
            rights = [p for p in (line.find(d, end) for d in DELIMITERS) if p >= 0]
            right = min(rights + [spans[i + 1][0] if i + 1 < len(spans) else len(line)])
            token, before, after = line[start:end], line[left:start], line[end:right]
            kind = "fill" if what == "fill" else _kind(token, before, after)
            prefix = blanked[:marks[i]]
            sentence = prefix[max(prefix.rfind(". ") + 2, prefix.rfind("| ") + 2, 0):]
            citations.append(Citation(doc, number, start + 1, kind, token, before, after, blanked, section, sentence))
    return citations, skipped


# -- the registry ------------------------------------------------------------------------

ENTRY_KEYS = {"id", "note", "anchor", "resolve", "fact"}
ANCHOR_KEYS = {"doc", "before", "after", "line", "section", "kind", "sentence"}
FACT_ANCHOR_KEYS = {"doc", "line", "section"}
FACT_AS = {"int", "ints", "str"}
RESOLVERS: dict[str, tuple[set[str], set[str]]] = {  # type -> (required, optional) keys besides "type"
    "blob_sha256": ({"repo", "ref", "path"}, {"decompress"}),
    "python": ({"repo", "ref", "call"}, {"args", "kwargs"}),
    "calibration_fingerprint": ({"repo", "ref", "script", "bank", "questions"}, {"k", "per_template"}),
    "commit": ({"repo", "ref"}, set()),
    "json_hash": ({"repo", "ref", "path", "pointer", "regex"}, set()),
    "mention": (set(), set()),
    "historical": (set(), set()),  # cited for the record only (change-log entry 126): reported, never resolved
}
VALUE_RESOLVERS: dict[str, tuple[set[str], set[str]]] = {
    "python_value": ({"repo", "ref"}, {"attr", "call", "args", "kwargs", "select"}),  # exactly one of attr, call
    "json_value": ({"repo", "ref", "path", "pointer"}, {"select"}),
}
SELECT = {"len", "values", "keys"}


@dataclass
class Entry:
    id: str
    anchor: dict[str, str]
    resolve: dict[str, Any]
    note: str = ""
    fact: dict[str, Any] | None = None

    def matches(self, c: Citation) -> bool:
        if "doc" in self.anchor and self.anchor["doc"] != c.doc:
            return False
        fields = {"before": c.before, "after": c.after, "line": c.blanked, "section": c.section, "kind": c.kind,
                  "sentence": c.sentence}
        return all(re.search(self.anchor[key], text) for key, text in fields.items() if key in self.anchor)


def _value_like(text: str) -> bool:
    return any(any(ch.isdigit() for ch in m.group()) for m in HEX.finditer(text))


def _value_group(pattern: str) -> str | None:
    """The source of the pattern's ``(?P<value>...)`` group, or None."""
    start = pattern.find("(?P<value>")
    if start < 0:
        return None
    depth, i = 0, start
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\":
            i += 2
            continue
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0:
            return pattern[start + len("(?P<value>"):i]
        i += 1
    return None


def _names_a_literal(group: str) -> bool:
    """Whether a value group spells part of a value (a digit or letter outside classes, escapes and counts)."""
    bare = re.sub(r"\\.", "", group)
    bare = re.sub(r"\[(?:\\.|[^\]])*\]", "", bare)
    bare = re.sub(r"\{\d+(?:,\d*)?\}|\(\?[:=!]", "", bare)
    return any(ch.isalnum() for ch in bare)


def _check_resolver(rid: str, resolve: dict[str, Any], table: dict[str, tuple[set[str], set[str]]], what: str) -> None:
    kind = resolve.get("type")
    if kind not in table:
        raise RegistryError(f"{rid}: {what} resolve type must be one of {sorted(table)}")
    required, optional = table[kind]
    keys = set(resolve) - {"type"}
    if required - keys or keys - required - optional:
        raise RegistryError(f"{rid}: a {kind} resolver takes {sorted(required)} and optionally {sorted(optional)}")
    if kind == "python_value" and len(keys & {"attr", "call"}) != 1:
        raise RegistryError(f"{rid}: a python_value resolver takes exactly one of attr and call")
    if resolve.get("select", "len") not in SELECT:
        raise RegistryError(f"{rid}: select is one of {sorted(SELECT)}")
    if kind == "json_hash":
        _check_hash_regex(rid, resolve["regex"])


def _check_hash_regex(rid: str, regex: Any) -> None:
    """A json_hash regex has one group, ``(?P<value>...)``, that spells no part of the hash it extracts."""
    try:
        compiled = re.compile(regex)
    except (re.error, TypeError) as exc:
        raise RegistryError(f"{rid}: regex: {exc}") from exc
    group = _value_group(regex)
    if compiled.groups != 1 or group is None or _names_a_literal(group):
        raise RegistryError(f"{rid}: the regex needs exactly one group, (?P<value>...), that names no literal digit or letter")
    if _value_like(regex):  # a lookahead outside the group could still spell the expected hash
        raise RegistryError(f"{rid}: the regex must name a pattern, never a hash value: {regex!r}")


def _load_fact(rid: str, raw: dict[str, Any]) -> Entry:
    fact, anchor, resolve = raw["fact"], raw.get("anchor", {}), raw.get("resolve")
    if not isinstance(fact, dict) or set(fact) != {"pattern", "as"} or fact["as"] not in FACT_AS:
        raise RegistryError(f"{rid}: a fact is {{pattern, as}}, with as one of {sorted(FACT_AS)}")
    group = _value_group(fact["pattern"])
    if group is None or _names_a_literal(group):
        raise RegistryError(f"{rid}: the pattern needs a (?P<value>...) group that names no literal digit or letter")
    try:
        compiled = re.compile(fact["pattern"])
    except re.error as exc:
        raise RegistryError(f"{rid}: pattern: {exc}") from exc
    if not isinstance(anchor, dict) or set(anchor) - FACT_ANCHOR_KEYS or not isinstance(resolve, dict):
        raise RegistryError(f"{rid}: a fact's anchor keys are {sorted(FACT_ANCHOR_KEYS)}, with a resolve")
    for key, pattern in anchor.items():
        if not isinstance(pattern, str) or _value_like(pattern):
            raise RegistryError(f"{rid}: anchor {key} must name a phrase, never a hash value: {pattern!r}")
    _check_resolver(rid, resolve, VALUE_RESOLVERS, "a fact's")
    return Entry(rid, anchor, resolve, raw.get("note", ""), {"pattern": compiled, "as": fact["as"]})


def load_registry(path: Path) -> list[Entry]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RegistryError(f"{path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema") != "freeze-registry/1" or set(data) - {"schema", "note", "entries"}:
        raise RegistryError(f"{path}: not a freeze-registry/1 object")
    entries, seen = [], set()
    for raw in data.get("entries", []):
        rid = raw.get("id") if isinstance(raw, dict) else None
        if not isinstance(rid, str) or rid in seen:
            raise RegistryError(f"{path}: an entry without a unique id: {raw!r}")
        seen.add(rid)
        if set(raw) - ENTRY_KEYS:
            raise RegistryError(f"{rid}: keys must be {sorted(ENTRY_KEYS)} (no expected values)")
        if "fact" in raw:
            entries.append(_load_fact(rid, raw))
            continue
        if not isinstance(raw.get("anchor"), dict) or not isinstance(raw.get("resolve"), dict):
            raise RegistryError(f"{rid}: keys must be {sorted(ENTRY_KEYS)} (no expected values), with anchor and resolve")
        anchor, resolve = raw["anchor"], raw["resolve"]
        if set(anchor) - ANCHOR_KEYS or not set(anchor) & {"before", "after", "line"}:
            raise RegistryError(f"{rid}: anchor keys are {sorted(ANCHOR_KEYS)}, with before, after or line")
        for key, pattern in anchor.items():
            if not isinstance(pattern, str) or _value_like(pattern):
                raise RegistryError(f"{rid}: anchor {key} must name a phrase, never a hash value: {pattern!r}")
            try:
                re.compile(pattern)
            except re.error as exc:
                raise RegistryError(f"{rid}: anchor {key}: {exc}") from exc
        _check_resolver(rid, resolve, RESOLVERS, "a hash entry's")
        entries.append(Entry(rid, anchor, resolve, raw.get("note", "")))
    return entries


# -- git and resolvers -------------------------------------------------------------------

class Resolver:
    """Recomputes an artefact's value from the artefact itself. It is never given the cited value."""

    def __init__(self, repos: dict[str, Path], workdir: Path, python: str,
                 remote_shell: tuple[str, ...] | list[str] = REMOTE_SHELL) -> None:
        self.repos = {name: Path(p) for name, p in repos.items()}
        self.workdir = workdir
        self.python = python
        self.remote_shell = list(remote_shell)
        self.refs: dict[tuple[str, str], str] = {}
        self.remote: list[dict[str, Any]] = []
        self._trees: dict[tuple[str, str, tuple[str, ...]], Path] = {}
        self._fetched: dict[tuple[str, str], Path] = {}

    def git(self, repo: str, *args: str, binary: bool = False) -> subprocess.CompletedProcess:
        if repo not in self.repos:
            raise ResolveError(f"repo {repo!r} is not given (known: {', '.join(sorted(self.repos))})")
        return subprocess.run(["git", "-C", str(self.repos[repo]), *args], capture_output=True, text=not binary,
                              timeout=TIMEOUT, check=False)

    def commit_of(self, repo: str, ref: str) -> str:
        if (repo, ref) not in self.refs:
            done = self.git(repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
            if done.returncode != 0:
                raise ResolveError(f"ref {ref!r} is not in {repo}")
            self.refs[(repo, ref)] = done.stdout.strip()
        return self.refs[(repo, ref)]

    def blob(self, repo: str, ref: str, path: str) -> bytes:
        commit = self.commit_of(repo, ref)
        done = self.git(repo, "cat-file", "blob", f"{commit}:{path}", binary=True)
        if done.returncode != 0:
            raise ResolveError(f"{path} is not at {repo} {ref} ({commit[:10]})")
        return done.stdout

    def tree(self, repo: str, ref: str, paths: tuple[str, ...] = CODE_PATHS) -> Path:
        commit = self.commit_of(repo, ref)
        key = (repo, commit, paths)
        if key not in self._trees:
            listed = self.git(repo, "ls-tree", "--name-only", commit, "--", *paths)
            present = listed.stdout.split() if listed.returncode == 0 else []
            if not present:
                raise ResolveError(f"none of {', '.join(paths)} is at {repo} {ref} ({commit[:10]})")
            data = self.git(repo, "archive", "--format=tar", commit, "--", *present, binary=True)
            if data.returncode != 0:
                raise ResolveError(f"git archive failed at {repo} {ref}: {data.stderr.decode(errors='replace').strip()}")
            dest = Path(tempfile.mkdtemp(prefix="tree-", dir=self.workdir))
            with tarfile.open(fileobj=io.BytesIO(data.stdout)) as tar:
                tar.extractall(dest, filter="data")
            self._trees[key] = dest
        return self._trees[key]

    def _source(self, spec: dict[str, Any], what: str) -> str:
        if set(spec) != {"repo", "ref", "path"}:
            raise ResolveError(f"{what} needs repo, ref and path")
        return spec["path"]

    def fetch_remote(self, spec: Any) -> Path:
        """A file or directory on another host, copied read-only (ssh + tar) into the run's directory."""
        if not isinstance(spec, dict) or set(spec) != {"host", "path"}:
            raise ResolveError("$remote needs host and path")
        host, path = spec["host"], spec["path"]
        if not isinstance(host, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", host):
            raise ResolveError(f"$remote host {host!r} is not a plain host name")
        remote = PurePosixPath(path) if isinstance(path, str) else None
        if remote is None or not remote.is_absolute() or ".." in remote.parts or remote == remote.parent:
            raise ResolveError(f"$remote path {path!r} must be absolute, without '..'")
        if (host, path) not in self._fetched:
            command = f"tar -C {shlex.quote(str(remote.parent))} -cf - -- {shlex.quote(remote.name)}"
            try:
                done = subprocess.run([*self.remote_shell, host, command], capture_output=True, timeout=TIMEOUT, check=False)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise ResolveError(f"{host}:{path} could not be read: {exc}") from exc
            if done.returncode != 0 or not done.stdout:
                lines = [x for x in done.stderr.decode(errors="replace").splitlines() if x.strip()]
                raise ResolveError(f"{host}:{path} could not be read: {lines[-1] if lines else 'no output'}")
            dest = Path(tempfile.mkdtemp(prefix="remote-", dir=self.workdir))
            with tarfile.open(fileobj=io.BytesIO(done.stdout)) as tar:
                tar.extractall(dest, filter="data")
            local = dest / remote.name
            files = [local] if local.is_file() else sorted(f for f in local.rglob("*") if f.is_file())
            self.remote.append({"host": host, "path": path, "files": {
                (f.name if f == local else f.relative_to(local).as_posix()): hashlib.sha256(f.read_bytes()).hexdigest()
                for f in files}})
            self._fetched[(host, path)] = local
        return self._fetched[(host, path)]

    def materialise(self, value: Any) -> Any:
        """A registry argument as the function receives it: $tree, $dir, $file, $jsonl, $fs, $remote become paths."""
        if isinstance(value, list):
            return [self.materialise(v) for v in value]
        if not isinstance(value, dict):
            return value
        if len(value) != 1 or next(iter(value)) not in {"$tree", "$dir", "$file", "$jsonl", "$fs", "$remote"}:
            return {k: self.materialise(v) for k, v in value.items()}
        (tag, spec), = value.items()
        if tag == "$remote":
            return {"__path__": str(self.fetch_remote(spec))}
        if tag == "$fs":
            if not Path(spec).exists():
                raise ResolveError(f"{spec} does not exist on this machine")
            return {"__path__": str(spec)}
        if tag == "$tree":
            return {"__path__": str(self.tree(spec["repo"], spec["ref"]))}
        path = self._source(spec, tag)
        if tag == "$dir":
            return {"__path__": str(self.tree(spec["repo"], spec["ref"], (path,)) / path)}
        dest = Path(tempfile.mkdtemp(prefix="blob-", dir=self.workdir)) / Path(path).name
        dest.write_bytes(self.blob(spec["repo"], spec["ref"], path))
        return {"__jsonl__" if tag == "$jsonl" else "__path__": str(dest)}

    def run_python(self, resolve: dict[str, Any], mode: str) -> Any:
        tree = self.tree(resolve["repo"], resolve["ref"])
        if mode == "calibration":
            namespace = {"bank": self.materialise(resolve["bank"]), "questions": self.materialise(resolve["questions"]),
                         "k": resolve.get("k", 3), "per_template": resolve.get("per_template"), "size": 1000,
                         "out": {"__path__": tempfile.mkdtemp(prefix="calib-out-", dir=self.workdir)}}
            spec = {"tree": str(tree), "call": f"{resolve['script']}:fingerprint", "mode": mode, "namespace": namespace}
        else:
            target = resolve.get("call") or resolve["attr"]
            spec = {"tree": str(tree), "call": target, "mode": mode, "invoke": "call" in resolve,
                    "args": self.materialise(resolve.get("args", [])), "kwargs": self.materialise(resolve.get("kwargs", {}))}
        spec_path = Path(tempfile.mkdtemp(prefix="spec-", dir=self.workdir)) / "spec.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        try:
            done = subprocess.run([self.python, "-I", "-B", "-c", DRIVER, str(spec_path)], capture_output=True, text=True,
                                  cwd=tree, timeout=TIMEOUT, check=False)
        except subprocess.TimeoutExpired as exc:
            raise ResolveError(f"{spec['call']} took over {TIMEOUT} s") from exc
        results = [line[len(MARK):] for line in done.stdout.splitlines() if line.startswith(MARK)]
        if done.returncode != 0 or not results:
            lines = [line for line in done.stderr.strip().splitlines() if line.strip()]
            raise ResolveError(f"{spec['call']} failed: {lines[-1] if lines else 'no output'}")
        return json.loads(results[-1])

    def digest(self, resolve: dict[str, Any], mode: str) -> str:
        value = self.run_python(resolve, mode)
        if not isinstance(value, str) or not DIGEST.fullmatch(value):
            target = resolve.get("call") or resolve.get("script")
            raise ResolveError(f"{target} returned {str(value)[:40]!r}, not a lowercase hex digest")
        return value

    def json_field(self, repo: str, ref: str, path: str, pointer: str) -> Any:
        """The value at a ``/a/0/b`` pointer in the JSON file ``path`` at ``ref``."""
        try:
            value = json.loads(self.blob(repo, ref, path))
        except ValueError as exc:
            raise ResolveError(f"{path} is not JSON: {exc}") from exc
        for part in pointer.split("/")[1:]:
            try:
                value = value[int(part)] if isinstance(value, list) else value[part]
            except (KeyError, IndexError, ValueError, TypeError) as exc:
                raise ResolveError(f"{path} has no {pointer}") from exc
        return value

    def fact_value(self, resolve: dict[str, Any]) -> tuple[Any, str]:
        """(recomputed value, artefact description) for a fact resolver."""
        repo, ref = resolve["repo"], resolve["ref"]
        commit = self.commit_of(repo, ref)
        where = f"{repo} {commit[:10]}" if commit.startswith(ref) else f"{repo} {ref}@{commit[:10]}"
        if resolve["type"] == "json_value":
            value = self.json_field(repo, ref, resolve["path"], resolve["pointer"])
            where += f":{resolve['path']}#{resolve['pointer']}"
        else:
            value = self.run_python(resolve, "value")
            where += f" {resolve.get('call') or resolve['attr']}" + ("()" if "call" in resolve else "")
        select = resolve.get("select")
        try:
            if select == "len":
                value = len(value)
            elif select in ("values", "keys"):
                value = list(value.values() if select == "values" else value.keys())
        except (TypeError, AttributeError) as exc:
            raise ResolveError(f"select {select} does not apply to {str(value)[:40]!r}") from exc
        return value, where + (f" [{select}]" if select else "")

    def value(self, resolve: dict[str, Any]) -> tuple[str, str]:
        """(recomputed value, artefact description) for a hash resolver."""
        kind, repo, ref = resolve["type"], resolve["repo"], resolve["ref"]
        commit = self.commit_of(repo, ref)
        where = f"{repo} {commit[:10]}" if commit.startswith(ref) else f"{repo} {ref}@{commit[:10]}"
        if kind == "blob_sha256":
            data = self.blob(repo, ref, resolve["path"])
            if resolve.get("decompress") == "gzip":
                data = gzip.decompress(data)
            return hashlib.sha256(data).hexdigest(), f"{where}:{resolve['path']}"
        if kind == "json_hash":
            path, pointer = resolve["path"], resolve["pointer"]
            text = self.json_field(repo, ref, path, pointer)
            if not isinstance(text, str):
                raise ResolveError(f"{path} {pointer} is not a string")
            found = re.search(resolve["regex"], text)
            if not found:
                raise ResolveError(f"{path} {pointer} ({text[:60]!r}) does not match the regex")
            if not DIGEST.fullmatch(found.group("value")):
                raise ResolveError(f"{path} {pointer} gives {found.group('value')[:40]!r}, not a lowercase hex digest")
            return found.group("value"), f"{where}:{path}#{pointer}"
        if kind == "python":
            return self.digest(resolve, "call"), f"{where} {resolve['call']}()"
        return self.digest(resolve, "calibration"), f"{where} {resolve['script']} fingerprint()"

    def commit(self, token: str, resolve: dict[str, Any]) -> tuple[str, str, str]:
        """(status, full commit or '', why) for a cited commit: it must exist and be reachable from the ref."""
        repo, ref = resolve["repo"], resolve["ref"]
        tip = self.commit_of(repo, ref)
        done = self.git(repo, "rev-parse", "--verify", f"{token}^{{commit}}")
        if done.returncode != 0:
            if "ambiguous" in done.stderr:
                raise ResolveError(f"{token} is ambiguous in {repo}")
            return "mismatch", "", f"no such commit in {repo} (is the clone fetched?)"
        full = done.stdout.strip()
        if self.git(repo, "merge-base", "--is-ancestor", full, tip).returncode != 0:
            on = self.git(repo, "branch", "-a", "--contains", full).stdout.split("\n")
            names = [b.strip(" *+") for b in on if b.strip()][:5]
            return "mismatch", full, f"in {repo} but not reachable from {ref}; on {', '.join(names) or 'no branch'}"
        behind = int(self.git(repo, "rev-list", "--count", f"{full}..{tip}").stdout.strip() or 0)
        place = "the tip" if behind == 0 else f"{behind} commit{'s' if behind != 1 else ''} behind the tip"
        return "match", full, f"on {ref} ({tip[:10]}), {place}"

    def hint(self, token: str) -> str:
        """Where a short unregistered token is a commit (a hint for the registry's author, never a match)."""
        if len(token) > 40:
            return ""
        found = [repo for repo in sorted(self.repos)
                 if self.git(repo, "rev-parse", "--verify", "--quiet", f"{token}^{{commit}}").returncode == 0]
        return f"; it is a commit in {', '.join(found)}" if found else ""


MARK = "FREEZE-RESULT "
DRIVER = r'''
import importlib, importlib.util, json, sys
from argparse import Namespace
from pathlib import Path

spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
tree = Path(spec["tree"]).resolve()
sys.path.insert(0, str(tree))


def arg(v):
    if isinstance(v, list):
        return [arg(x) for x in v]
    if isinstance(v, dict) and set(v) == {"__jsonl__"}:
        return [json.loads(x) for x in Path(v["__jsonl__"]).read_text(encoding="utf-8").splitlines() if x.strip()]
    if isinstance(v, dict) and set(v) == {"__path__"}:
        return Path(v["__path__"])
    if isinstance(v, dict):
        return {k: arg(x) for k, x in v.items()}
    return v


def inside(module):
    f = getattr(module, "__file__", None)
    return bool(f) and Path(f).resolve().is_relative_to(tree)


name, _, func = spec["call"].partition(":")
if name.endswith(".py"):
    mspec = importlib.util.spec_from_file_location("freeze_target", tree / name)
    module = importlib.util.module_from_spec(mspec)
    sys.modules["freeze_target"] = module
    mspec.loader.exec_module(module)
else:
    module = importlib.import_module(name)
if not inside(module):
    raise SystemExit(f"{name} loaded from {getattr(module, '__file__', None)}, outside the extracted tree")
if spec["mode"] == "calibration":
    ns = Namespace(**{k: arg(v) for k, v in spec["namespace"].items()})
    _, _, confs, _ = module.setup(ns)
    value = getattr(module, func)(ns, confs)
elif spec["mode"] == "value" and not spec["invoke"]:
    value = getattr(module, func)
else:
    value = getattr(module, func)(*arg(spec["args"]), **arg(spec["kwargs"]))
foreign = [n for n, m in list(sys.modules.items()) if (n == "harness" or n.startswith("harness.")) and not inside(m)]
if foreign:
    raise SystemExit(f"harness modules loaded from outside the extracted tree: {', '.join(sorted(foreign))}")
print("FREEZE-RESULT " + json.dumps(value, default=lambda o: sorted(o) if isinstance(o, (set, frozenset)) else list(o)))
'''


# -- the check ---------------------------------------------------------------------------

@dataclass
class Result:
    citation: Citation
    status: str
    registry_id: str | None = None
    artefact: str = ""
    recomputed: str = ""
    why: str = ""


GRID_SPEC = "data/runs/dev-grid/grid.json"
CHECK_CONTRACT = "paper/s01-check-contract.md"
FROZEN_CONFIG_PIN = "data/freeze/frozen-config.json"
SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass
class Binding:
    id: str
    pinned_in: str  # where the pin is held, at the experiments' bind ref
    artefact: str  # what the pinned value must be the sha256 of
    status: str
    pinned: str = ""
    recomputed: str = ""
    why: str = ""


def _binding(bid: str, pinned_in: str, artefact: str, read_pin, read_artefact) -> Binding:
    """One binding: the pinned value, read first, then the artefact's sha256, recomputed; equal or not, exactly."""
    try:
        pinned = read_pin()
    except (ResolveError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return Binding(bid, pinned_in, artefact, "unresolvable", why=f"the pin cannot be read: {exc}")
    try:
        value = hashlib.sha256(read_artefact()).hexdigest()
    except (ResolveError, ValueError, KeyError, TypeError) as exc:
        return Binding(bid, pinned_in, artefact, "unresolvable", str(pinned), why=f"the artefact cannot be read: {exc}")
    if isinstance(pinned, str) and (FILL.fullmatch(pinned) or BARE_FILL.fullmatch(pinned)):
        return Binding(bid, pinned_in, artefact, "unresolved", pinned, value, "a placeholder; recomputed is the current value")
    if not (isinstance(pinned, str) and SHA256.fullmatch(pinned)):
        return Binding(bid, pinned_in, artefact, "unresolvable", str(pinned), value, "the pin is not a sha256")
    return Binding(bid, pinned_in, artefact, "match" if pinned == value else "mismatch", pinned, value)


def check_bindings(resolver: "Resolver", science_commit: str, ref: str) -> list[Binding]:
    """The pins the experiments hold at ``ref`` of artefacts elsewhere, each against its artefact."""
    def grid_pin():
        return json.loads(resolver.blob("experiments", ref, GRID_SPEC))["code"]["check_contract_sha256"]

    def config_pin():
        return json.loads(resolver.blob("experiments", ref, FROZEN_CONFIG_PIN))

    def config_path():
        path = config_pin()["path"]
        if not isinstance(path, str) or not path:
            raise ValueError("the pin names no config path")
        return path

    out = [_binding("check-contract", f"experiments {ref}:{GRID_SPEC} code.check_contract_sha256",
                    f"science {science_commit[:12]}:{CHECK_CONTRACT}", grid_pin,
                    lambda: resolver.blob("science", science_commit, CHECK_CONTRACT))]
    try:
        named = config_path()
    except (ResolveError, ValueError, KeyError, TypeError, AttributeError):
        named = "(the config the pin names)"
    out.append(_binding("frozen-config", f"experiments {ref}:{FROZEN_CONFIG_PIN} sha256", f"experiments {ref}:{named}",
                        lambda: config_pin()["sha256"], lambda: resolver.blob("experiments", ref, config_path())))
    out += fault_rule_bindings(resolver, ref, config_pin)
    return out


def fault_rule_bindings(resolver: Resolver, ref: str, config_pin) -> list[Binding]:
    """The test runs' fault rule (change-log entries 148 and 149): the frozen config's pin of the fault-rule file
    (``fault_rule``), and the rule file's own pins of what it binds in the repository: the dev baselines it took its
    rates from, each model's probe set and each store copy's probe set held in git (an a15 copy's set stays in the
    private tree, so run_probe checks it there by the same sha256). A rule file that cannot be read leaves only its
    own binding, unresolvable."""
    def rule_path() -> str:
        path = (config_pin().get("fault_rule") or {}).get("path")
        if not isinstance(path, str) or not path:
            raise ValueError("the frozen config pins no fault-rule file")
        return path

    try:
        named = rule_path()
    except (ResolveError, ValueError, KeyError, TypeError, AttributeError):
        named = "(the fault-rule file the pin names)"
    out = [_binding("fault-rule", f"experiments {ref}:{FROZEN_CONFIG_PIN} fault_rule.sha256", f"experiments {ref}:{named}",
                    lambda: (config_pin().get("fault_rule") or {})["sha256"],
                    lambda: resolver.blob("experiments", ref, rule_path()))]
    try:
        rule = json.loads(resolver.blob("experiments", ref, named))
    except (ResolveError, ValueError, KeyError, TypeError):
        return out
    pins = [("fault-dev-baselines", "dev_rates.sha256", (rule.get("dev_rates") or {}).get("sha256"),
             (rule.get("dev_rates") or {}).get("file"))]
    pins += [(f"fault-probe-set:{m}", f"probe_sets.{m}.sha256", e.get("sha256"), e.get("file"))
             for m, e in sorted((rule.get("probe_sets") or {}).items())]
    for copy, c in sorted(((rule.get("stores") or {}).get("copies") or {}).items()):
        ps = c.get("probe_set") if isinstance(c, dict) else None
        if isinstance(ps, dict) and str(ps.get("path", "")).startswith("data/"):
            pins.append((f"store-probe-set:{copy}", f"stores.copies.{copy}.probe_set.sha256", ps.get("sha256"),
                         ps["path"]))
    for bid, field_name, pinned, path in pins:
        out.append(_binding(bid, f"experiments {ref}:{named} {field_name}", f"experiments {ref}:{path}",
                            lambda pinned=pinned: pinned,
                            lambda path=path: resolver.blob("experiments", ref, path)))
    return out


@dataclass
class Report:
    science_commit: str
    documents: dict[str, str]
    registry_path: str
    registry_sha256: str
    refs: list[dict[str, str]]
    results: list[Result]
    skipped: list[dict[str, Any]] = field(default_factory=list)
    unused: list[str] = field(default_factory=list)  # registry entries no citation reached
    remote: list[dict[str, Any]] = field(default_factory=list)  # inputs read from other hosts, with their sha256
    bindings: list[Binding] | None = None  # None: not checked (no --bind-ref)
    bind_ref: str | None = None

    def counts(self) -> dict[str, int]:
        statuses = [r.status for r in self.results] + [b.status for b in self.bindings or []]
        return {s: statuses.count(s) for s in STATUSES}

    def exit_code(self, allow_fill: bool) -> int:
        counts = self.counts()
        failed = counts["mismatch"] + counts["unresolvable"] + (0 if allow_fill else counts["unresolved"])
        return 1 if failed else 0


def check_one(c: Citation, entries: list[Entry], resolver: Resolver) -> Result:
    named = [e for e in entries if e.matches(c)]
    if c.kind == "fill":
        named = [e for e in named if e.resolve["type"] != "historical"]  # a placeholder is never history
        if any(e.resolve["type"] == "mention" for e in named):
            mention = next(e for e in named if e.resolve["type"] == "mention")
            return Result(c, "ignored" if len(named) == 1 else "unresolved", mention.id,
                          why=mention.note or "the registry marks this as prose about the marker")
        artefacts = [e for e in named if e.resolve["type"] != "commit"]
        if len(named) == 1 and artefacts:
            try:
                value, where = resolver.value(artefacts[0].resolve)
                return Result(c, "unresolved", artefacts[0].id, where, value, "a placeholder; recomputed is the current value")
            except ResolveError as exc:
                return Result(c, "unresolved", artefacts[0].id, why=f"a placeholder; the current value: {exc}")
        if len(named) == 1:  # a commit slot: nothing to recompute until a commit is cited
            return Result(c, "unresolved", named[0].id, f"{named[0].resolve['repo']} commit on {named[0].resolve['ref']}",
                          why="a placeholder")
        return Result(c, "unresolved", why="a placeholder" if not named else
                      f"a placeholder; ambiguous: entries {', '.join(e.id for e in named)} name this phrase")
    named = [e for e in named if e.resolve["type"] != "mention"]
    if not named:
        return Result(c, "unresolvable", why="no registry entry names this phrase" + resolver.hint(c.token))
    if len(named) > 1:
        return Result(c, "unresolvable", why=f"ambiguous: entries {', '.join(e.id for e in named)} name this phrase")
    entry = named[0]
    if entry.resolve["type"] == "historical":
        return Result(c, "historical", entry.id, why=entry.note or "cited for the record only; bound to nothing")
    try:
        if entry.resolve["type"] == "commit":
            status, full, why = resolver.commit(c.token, entry.resolve)
            return Result(c, status, entry.id, f"{entry.resolve['repo']} commit on {entry.resolve['ref']}", full, why)
        value, where = resolver.value(entry.resolve)
    except ResolveError as exc:
        return Result(c, "unresolvable", entry.id, why=str(exc))
    # The only place the cited value is used: after the artefact's value has been recomputed.
    status = "match" if value.startswith(c.token) else "mismatch"
    return Result(c, status, entry.id, where, value)


def extract_facts(doc: str, text: str, facts: list[Entry]) -> list[tuple[Citation, Entry]]:
    """Every value a fact entry's pattern reads in ``text``, where its anchor allows."""
    found = []
    section = ""
    for number, line in enumerate(text.splitlines(), start=1):
        heading = HEADING.match(line)
        if heading:
            section = heading.group(1)
        for entry in facts:
            a = entry.anchor
            if ("doc" in a and a["doc"] != doc) or ("line" in a and not re.search(a["line"], line)) or (
                    "section" in a and not re.search(a["section"], section)):
                continue
            for m in entry.fact["pattern"].finditer(line):
                start, end = m.span("value")
                found.append((Citation(doc, number, start + 1, "fact", m.group("value"), line[:start], line[end:], line,
                                       section), entry))
    return found


def _read_cited(text: str, as_: str) -> Any:
    if as_ == "int":
        return int(re.sub(r"[,\s_\u2009\u202f]", "", text))
    if as_ == "ints":
        return [int(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\d|\d", text)]
    return text.strip()


def _is(value: Any, as_: str) -> bool:
    if as_ == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if as_ == "ints":
        return isinstance(value, list) and all(isinstance(v, int) and not isinstance(v, bool) for v in value)
    return isinstance(value, str)


def check_fact(c: Citation, entry: Entry, resolver: Resolver) -> Result:
    as_ = entry.fact["as"]
    try:
        value, where = resolver.fact_value(entry.resolve)
    except ResolveError as exc:
        return Result(c, "unresolvable", entry.id, why=str(exc))
    shown = json.dumps(value, ensure_ascii=False)
    if not _is(value, as_):
        return Result(c, "unresolvable", entry.id, where, shown, f"the artefact's value is not {as_}")
    try:
        cited = _read_cited(c.token, as_)
    except ValueError:
        return Result(c, "unresolvable", entry.id, where, shown, f"the cited text cannot be read as {as_}")
    # As for hashes, the cited value is read only now, after the artefact's value.
    return Result(c, "match" if cited == value else "mismatch", entry.id, where, shown)


def run_check(science_repo: Path, commit: str, registry_path: Path, repos: dict[str, Path],
              docs: tuple[str, ...] = DOCS, python: str | None = None,
              remote_shell: tuple[str, ...] | list[str] = REMOTE_SHELL, bind_ref: str | None = None) -> Report:
    entries = load_registry(registry_path)
    facts = [e for e in entries if e.fact is not None]
    phrases = [e for e in entries if e.fact is None]
    repos = {"science": Path(science_repo), **repos}
    with tempfile.TemporaryDirectory(prefix="freeze-check-") as tmp:
        resolver = Resolver(repos, Path(tmp), python or sys.executable, remote_shell)
        try:
            science_commit = resolver.commit_of("science", commit)
        except ResolveError as exc:
            raise RegistryError(f"the science repo: {exc}") from exc
        citations, skipped, documents, cited_facts = [], [], {}, []
        for doc in docs:
            try:
                text = resolver.blob("science", commit, doc).decode("utf-8")
            except ResolveError as exc:
                raise RegistryError(f"the document: {exc}") from exc
            documents[doc] = resolver.git("science", "rev-parse", f"{science_commit}:{doc}").stdout.strip()
            found, dropped = extract(doc, text)
            citations += found
            skipped += dropped
            cited_facts += extract_facts(doc, text, facts)
        results = [check_one(c, phrases, resolver) for c in citations]
        results += [check_fact(c, entry, resolver) for c, entry in cited_facts]
        results.sort(key=lambda r: (docs.index(r.citation.doc), r.citation.line, r.citation.col))
        used = {e.id for e in phrases for c in citations if e.matches(c)} | {e.id for _, e in cited_facts}
        bindings = check_bindings(resolver, science_commit, bind_ref) if bind_ref is not None else None
        refs = [{"repo": repo, "ref": ref, "commit": sha} for (repo, ref), sha in resolver.refs.items()]
    registry_sha = hashlib.sha256(Path(registry_path).read_bytes()).hexdigest()
    return Report(science_commit, documents, _display_path(registry_path), registry_sha, refs, results, skipped,
                  [e.id for e in entries if e.id not in used], resolver.remote, bindings, bind_ref)


def _display_path(path: Path) -> str:
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name


# -- output ------------------------------------------------------------------------------

def _short(value: str, n: int = 16) -> str:
    return value if len(value) <= n else value[:n] + "…"


def _cell(text: str) -> str:
    return text.replace("`", "").replace("**", "").replace("|", "\\|").replace("\n", " ").strip()


def _context(c: Citation, width: int = 70) -> str:
    before, after = c.before.strip(), c.after.strip()
    if len(before) > width:
        before = "…" + before[-width:]
    if len(after) > width // 2:
        after = after[:width // 2] + "…"
    return f"{before} «{_short(c.token, 20)}» {after}".strip()


def to_markdown(report: Report, allow_fill: bool) -> str:
    counts = report.counts()
    out = [f"# Freeze-binding check, {datetime.now(UTC):%Y-%m-%d %H:%M} UTC", "",
           f"- **Science commit:** `{report.science_commit}`"]
    out += [f"- **{doc}:** blob `{sha}`" for doc, sha in report.documents.items()]
    out += [f"- **Registry:** `{report.registry_path}`, sha256 `{report.registry_sha256}`",
            "- **Refs resolved:** " + ("; ".join(f"{r['repo']} `{r['ref']}` = `{r['commit'][:12]}`" for r in report.refs) or "none"),
            "- **Remote inputs (read-only):** " + ("; ".join(
                f"{r['host']}:{r['path']} (" + ", ".join(f"{name} `{sha[:16]}…`" for name, sha in r["files"].items()) + ")"
                for r in report.remote) or "none"),
            "- **Registry entries no citation reached:** " + (", ".join(report.unused) or "none"),
            "- **Counts:** " + ", ".join(f"{s} {counts[s]}" for s in STATUSES),
            f"- **Exit:** {report.exit_code(allow_fill)}" + (" (--allow-fill)" if allow_fill else ""), "",
            "| # | Where | Kind | Cited | Status | Registry entry | Artefact | Recomputed | Why | Context |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(report.results, start=1):
        c = r.citation
        out.append(f"| {i} | {c.doc.removeprefix('paper/')}:{c.line} | {c.kind} | {_cell(_short(c.token, 24))} | "
                   f"**{r.status}** | {r.registry_id or '–'} | {_cell(r.artefact) or '–'} | "
                   f"{_cell(_short(r.recomputed, 40 if c.kind == 'fact' else 16)) or '–'} | "
                   f"{_cell(r.why) or '–'} | {_cell(_context(c))} |")
    out += ["", f"Bindings (experiments at {report.bind_ref}):" if report.bindings is not None
            else "Bindings: not checked (--no-bindings).", ""]
    if report.bindings is not None:
        out += ["| Binding | Status | Pinned in | Pinned | Artefact | Recomputed | Why |", "|---|---|---|---|---|---|---|"]
        out += [f"| {b.id} | **{b.status}** | {_cell(b.pinned_in)} | {_cell(_short(b.pinned, 16)) or '–'} | "
                f"{_cell(b.artefact)} | {_cell(_short(b.recomputed, 16)) or '–'} | {_cell(b.why) or '–'} |"
                for b in report.bindings]
    if report.skipped:
        out += ["", "Hex-like tokens not counted as citations:", ""]
        out += [f"- {s['doc']}:{s['line']} `{s['token']}`: {s['why']}" for s in report.skipped]
    return "\n".join(out) + "\n"


def to_json(report: Report, allow_fill: bool) -> dict[str, Any]:
    return {"schema": "freeze-binding-check/1", "generated": datetime.now(UTC).isoformat(timespec="seconds"),
            "science": {"commit": report.science_commit, "documents": report.documents},
            "registry": {"path": report.registry_path, "sha256": report.registry_sha256}, "refs": report.refs,
            "counts": report.counts(), "allow_fill": allow_fill, "exit": report.exit_code(allow_fill),
            "entries": [{"doc": r.citation.doc, "line": r.citation.line, "col": r.citation.col, "kind": r.citation.kind,
                         "cited": r.citation.token, "status": r.status, "registry_id": r.registry_id,
                         "artefact": r.artefact, "recomputed": r.recomputed, "why": r.why,
                         "section": r.citation.section, "before": r.citation.before, "after": r.citation.after}
                        for r in report.results],
            "skipped": report.skipped, "unused_registry_entries": report.unused, "remote_inputs": report.remote,
            "bind_ref": report.bind_ref,
            "bindings": None if report.bindings is None else [dataclasses.asdict(b) for b in report.bindings]}


# -- command line ------------------------------------------------------------------------

def _default_science_repo() -> Path:
    if os.environ.get("WWW2027_SCIENCE_REPO"):
        return Path(os.environ["WWW2027_SCIENCE_REPO"])
    done = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                          capture_output=True, text=True, check=False)
    common = Path(done.stdout.strip()) if done.returncode == 0 else ROOT / ".git"
    return common.parent.parent / "science-repo"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--science-repo", type=Path, default=None, help="default: the science-repo clone beside this repo")
    ap.add_argument("--commit", default=DEFAULT_COMMIT, help=f"the science-repo commit to read (default {DEFAULT_COMMIT})")
    ap.add_argument("--doc", action="append", dest="docs", help=f"a document to check (default: {', '.join(DOCS)})")
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument("--repo", action="append", default=[], metavar="NAME=DIR",
                    help="an artefact repo the registry names (default: experiments = this repo)")
    ap.add_argument("--md", type=Path, help="write the markdown table here (default: stdout)")
    ap.add_argument("--json", type=Path, help="write the JSON report here")
    ap.add_argument("--allow-fill", action="store_true", help="exit 0 despite [FILL] placeholders")
    ap.add_argument("--python", default=sys.executable, help="the interpreter for code-running resolvers")
    ap.add_argument("--bind-ref", default="HEAD", help="the experiments ref whose pins are checked (the grid's check "
                                                       "contract, the frozen config's pin); default HEAD, at the freeze "
                                                       "the freeze tag")
    ap.add_argument("--no-bindings", action="store_true", help="skip the bindings (the report says so)")
    args = ap.parse_args(argv)
    repos: dict[str, Path] = {"experiments": ROOT}
    for item in args.repo:
        name, sep, path = item.partition("=")
        if not sep or not name:
            ap.error(f"--repo takes NAME=DIR, got {item!r}")
        repos[name] = Path(path)
    try:
        report = run_check(args.science_repo or _default_science_repo(), args.commit, args.registry, repos,
                           tuple(args.docs or DOCS), args.python,
                           bind_ref=None if args.no_bindings else args.bind_ref)
    except RegistryError as exc:
        print(f"freeze_binding_check: {exc}", file=sys.stderr)
        return 2
    md = to_markdown(report, args.allow_fill)
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(md, encoding="utf-8")
    else:
        sys.stdout.write(md)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(to_json(report, args.allow_fill), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    counts = report.counts()
    print("freeze_binding_check: " + ", ".join(f"{s} {counts[s]}" for s in STATUSES), file=sys.stderr)
    return report.exit_code(args.allow_fill)


if __name__ == "__main__":
    sys.exit(main())
