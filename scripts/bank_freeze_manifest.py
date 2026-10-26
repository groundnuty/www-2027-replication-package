#!/usr/bin/env python3
"""The bank job's freeze manifest, made on the VM where git is (lead, 21 Oct: no git clone on SiteA, whose project
space is group-readable while the repository's objects hold the held-out blobs).

    uv run python scripts/bank_freeze_manifest.py --freeze-repo DIR --freeze-tag TAG --prereg-repo SCI \
        --prereg-commit SHA --signed-off-by NAME --out FREEZE-MANIFEST.json

The frozen pre-registration is verified by the test grid's own gate (scripts/bank_replay.py frozen_preregistration:
the exact line 'Status: frozen' at --prereg-commit, that commit on the science origin's main as fetched now), and its
bytes are read from that commit; the freeze tag is read from --freeze-repo's origin as it is now
(bankrun.make_freeze_manifest: the tag's commit, its tree digest from git's objects, its config's sha256). The manifest
goes to SiteA with the deploy, and the job's gate checks the deploy against it (bank_replay.py --freeze-manifest).
Exit 0 written, 2 refused.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.checkloop.bankrun import PREREG_PATH, BankReplayError, make_freeze_manifest  # noqa: E402


def _replay_gate():
    spec = importlib.util.spec_from_file_location("bank_replay_gate", ROOT / "scripts" / "bank_replay.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["bank_replay_gate"] = module
    spec.loader.exec_module(module)
    return module.frozen_preregistration


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--freeze-repo", type=Path, required=True, help="a git clone of this repository (its origin has the tag)")
    p.add_argument("--freeze-tag", required=True)
    p.add_argument("--prereg-repo", type=Path, required=True, help="a clone of the science repository")
    p.add_argument("--prereg-commit", required=True, help="the commit whose paper/preregistration.md says 'Status: frozen'")
    p.add_argument("--signed-off-by", required=True, help="who verified and signed the manifest off")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        prereg = _replay_gate()(args.prereg_repo, args.prereg_commit)
        shown = subprocess.run(["git", "-C", str(args.prereg_repo), "show", f"{prereg['commit']}:{PREREG_PATH}"],
                               capture_output=True, check=False)
        if shown.returncode != 0:
            raise BankReplayError(f"cannot read {PREREG_PATH} at {prereg['commit'][:12]}")
        manifest = make_freeze_manifest(args.freeze_repo, args.freeze_tag, prereg, shown.stdout, args.signed_off_by)
    except BankReplayError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out} (sha256 {hashlib.sha256(text.encode()).hexdigest()}: give it to the job as "
          f"--freeze-manifest-sha256): tag {manifest['freeze_tag']} at "
          f"{manifest['commit']}, tree {manifest['tree_sha256'][:12]}, config {manifest['config_sha256'][:12]}, "
          f"pre-registration {manifest['preregistration']['commit'][:12]} {manifest['preregistration']['sha256'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
