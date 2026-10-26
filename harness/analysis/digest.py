"""One code hash binds everything (consolidated plan D): the tree digest of harness/ and scripts/.

The digest is computed from file contents, not from git, so a clean copy without ``.git`` (the
certification run on run-node) gives the same value as the worktree. It is the sha256 of the
compact JSON list of ``[path, sha256 of the file]`` pairs, sorted by path, over every file under
``harness/`` and ``scripts/`` relative to the repository root. ``__pycache__`` directories and
``*.pyc`` files are excluded. Every results file records it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
TREES = ("harness", "scripts")


def code_files(root: Path = ROOT) -> list[str]:
    root = Path(root)
    found = []
    for tree in TREES:
        for path in (root / tree).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                found.append(path.relative_to(root).as_posix())
    return sorted(found)


def code_digest(root: Path = ROOT) -> str:
    """The tree digest of harness/ and scripts/ (plan D)."""
    root = Path(root)
    pairs = [[rel, hashlib.sha256((root / rel).read_bytes()).hexdigest()] for rel in code_files(root)]
    return hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode("utf-8")).hexdigest()
