"""The scorer digest (prereg section 3): the frozen scorer is cited as the main commit plus this sha256.

It covers a closed set of modules, ``SCORER_MODULES``: every scorer module and every in-repo module they import,
with the packages those imports run (tests/test_score_digest.py walks the imports and fails if the set and the walk
differ). ``scorer_digest()`` is the sha256 of the compact JSON ``{"modules": [...], "sources": [...]}``: the module
list in its fixed order and the sha256 of each file's text with line endings normalised to ``\\n``. It reads files,
not git, so a clean copy without ``.git`` gives the same value. This module itself is not in the set: it is the
instrument, and the commit cited beside the digest pins it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SCORER_MODULES = (
    "harness/__init__.py",
    "harness/score/__init__.py",
    "harness/score/gold.py",
    "harness/score/rules.py",
    "harness/score/scorer.py",
    "harness/terms.py",
)


def _normalised(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def module_digests(root: Path = ROOT) -> dict[str, str]:
    """Each module's normalised-source sha256 (to name what changed when the digest differs)."""
    return {rel: hashlib.sha256(_normalised(Path(root) / rel)).hexdigest() for rel in SCORER_MODULES}


def scorer_digest(root: Path = ROOT) -> str:
    """sha256 of the module list and each module's normalised source, in the listed order."""
    sources = module_digests(root)
    payload = {"modules": list(SCORER_MODULES), "sources": [sources[rel] for rel in SCORER_MODULES]}
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()
