"""Which file holds a graph's source of a given name: one rule for every builder (External review of ee6fd40, item 2).

A source is written as ``<name>.json`` (scripts/build_sources.py) and may also be kept as ``<name>.json.gz`` (the
committed form prompt assembly reads). Both builders resolve a name here, so they read the same source: the gzip form
when both files hold the same source, a refusal when they differ (a rebuilt plain file beside an older gzip one).
A source is identified by the sha256 of its canonical JSON (``content_sha256``), the same in either form: a fit
computation records it for each measured source, and both builders pass the render guard those of the sources they
render (External review of bebd077, item 2). Kept byte-identical on feat/sources and feat/prompt-conditions.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path


def read_json(path: Path) -> dict:
    raw = Path(path).read_bytes()
    return json.loads(gzip.decompress(raw) if Path(path).suffix == ".gz" else raw)


def resolve(src_dir: Path, name: str) -> Path | None:
    """The file of the source ``name`` in ``src_dir`` (``<name>.json.gz`` or ``<name>.json``), or None; refused when
    both exist and hold different sources."""
    plain, packed = Path(src_dir) / f"{name}.json", Path(src_dir) / f"{name}.json.gz"
    if plain.exists() and packed.exists():
        if read_json(plain) != read_json(packed):
            raise ValueError(f"{src_dir}: {plain.name} and {packed.name} hold different sources; remove the stale one "
                             "before building")
        return packed
    if packed.exists():
        return packed
    return plain if plain.exists() else None


def name_of(path: Path) -> str:
    """The name of the source a file holds by its name: ``census-t5`` for census-t5.json and census-t5.json.gz."""
    return Path(path).name.removesuffix(".gz").removesuffix(".json")


def content_sha256(path: Path) -> str:
    """sha256 of the source's canonical JSON (sorted keys, no spaces), whichever form the file has."""
    return hashlib.sha256(json.dumps(read_json(path), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def content_digests(paths: Iterable[Path]) -> dict[str, str]:
    """``content_sha256`` of each source file, by source name: what a builder passes the render guard."""
    return {name_of(p): content_sha256(p) for p in paths}
