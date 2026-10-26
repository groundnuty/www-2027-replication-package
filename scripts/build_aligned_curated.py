"""Build the aligned curated source from a curated source file and the reviewed SSSOM mapping (harness.sources.align).

    python -I scripts/build_aligned_curated.py --curated <curated.json> --curated-sha256 <hex> \\
        --mapping <reviewed .tsv> --curie-map <curie_map.yaml> --out <private dir>

The mapping is refused unless its sha256 is the reviewed file's (``REVIEWED_MAPPING_SHA256``; there is no override),
and the curated file unless it has the given sha256. Writes <out>/aligned-curated.json (mode 600 in a mode-700
directory), checked to load as a source, and prints its sha256, digest and counts only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from generated_from_files import _write_private, private_writes

from harness.sources.align import AlignmentRefused, align, read_mapping
from harness.sources.model import OntologySource

# The curator's reviewed mapping (change-log entry 98): the aligned source is built from this file only.
REVIEWED_MAPPING_SHA256 = "fbaac3af23d986bf292b0568e6812c46104f71a608ba0e6799d7d9473ab5651a"
NAME = "aligned-curated"


def build(curated: Path, curated_sha256: str, mapping: Path, curie_map: Path, out: Path) -> Path:
    """The aligned source from the reviewed mapping only: the mapping's sha256 is always checked against
    ``REVIEWED_MAPPING_SHA256``."""
    raw = curated.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != curated_sha256:
        raise AlignmentRefused(f"curated file sha256 {sha} is not the recorded one")
    rename, kinds, record = read_mapping(mapping, curie_map, REVIEWED_MAPPING_SHA256)
    aligned = align(json.loads(raw), rename, kinds, {**record, "curated_sha256": sha}, NAME)
    OntologySource.from_json(aligned)  # refuse anything that does not load as a source
    path = out / f"{NAME}.json"
    with private_writes(out):
        _write_private(path, (json.dumps(aligned, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                              + "\n").encode())
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--curated", type=Path, required=True)
    ap.add_argument("--curated-sha256", required=True)
    ap.add_argument("--mapping", type=Path, required=True)
    ap.add_argument("--curie-map", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    umask = os.umask(0o077)
    try:
        path = build(args.curated, args.curated_sha256, args.mapping, args.curie_map, args.out)
        src = OntologySource.from_json(json.loads(path.read_bytes()))
        rec = src.provenance["alignment"]
        print(json.dumps({"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "digest": src.digest(), "classes": len(src.classes), "properties": len(src.properties),
                          **{k: rec[k] for k in ("exact_matches", "renamed_terms", "source_terms_not_mapped", "axioms",
                                                 "axioms_touched", "mapping_sha256", "curated_sha256")}},
                         sort_keys=True))
        return 0
    except AlignmentRefused as exc:  # its message is hashes and counts only
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - a traceback can quote private terms: it goes to a private file
        log, where = args.out / "error.log", ""
        try:
            _write_private(log, traceback.format_exc().encode())
            where = "; traceback in error.log in the --out directory"
        except OSError as werr:
            where = f"; the traceback could not be written ({type(werr).__name__})"
        print(f"failed: {type(exc).__name__}{where}", file=sys.stderr)
        return 1
    finally:
        os.umask(umask)


if __name__ == "__main__":
    sys.exit(main())
