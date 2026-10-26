#!/usr/bin/env python3
"""Write COMMIT and TREE at the code root, for a deployed copy without git (runner review).

    uv run python scripts/write_tree_digest.py [--root DIR]

Run in a clean checkout, before copying the code to the run host (the recipe is in
harness/agent/replay_check.py). COMMIT gets git HEAD; TREE gets the tree digest of the code
files (replay_check.CODE_PATHS) bound to that commit, {"schema": "code-tree/1", "commit",
"tree_sha256"}, so a COMMIT left from another commit is refused (F1 review, item 10). The runners
and the replay-check gate recompute the digest from the files on disk and accept the code only
while it is TREE's and COMMIT is the commit TREE was written for. Refuses, writing
nothing, when --root is not git's checkout root, when the code paths have uncommitted changes
or untracked files, or when git cannot say.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Allow running as a plain script from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.agent.replay_check import (  # noqa: E402
    CODE_PATHS,
    CODE_ROOT,
    tree_file_text,
    tree_sha256,
)


def _git(root: Path, *args: str) -> str | None:
    done = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=10, check=False)
    return done.stdout.strip() if done.returncode == 0 else None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", type=Path, default=CODE_ROOT, help="the checkout (default: this code's root)")
    args = p.parse_args(argv)
    root = args.root.resolve()
    top = _git(root, "rev-parse", "--show-toplevel")
    if top is not None and Path(top).resolve() != root:  # a subdirectory would digest an incomplete tree
        print(f"error: {root} is not the checkout's root ({top}); pass --root {top}", file=sys.stderr)
        return 1
    head = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain", "--untracked-files=all", "--", *CODE_PATHS)
    if head is None or status is None:
        print(f"error: {root} is not a git checkout git can read", file=sys.stderr)
        return 1
    if status:
        print(f"error: {root} has uncommitted changes to the code; commit them first:\n{status}", file=sys.stderr)
        return 1
    tree = tree_sha256(root)
    (root / "COMMIT").write_text(head + "\n", encoding="utf-8")
    (root / "TREE").write_text(tree_file_text(head, tree), encoding="utf-8")
    print(f"COMMIT {head}\nTREE {tree}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
