"""Change-log entry 92's per-layer overlap of each generated draft with the census and the curated source.

    uv run python scripts/overlap_reading.py --graph text2sparql-ck26
    python -I scripts/overlap_reading.py --graph a15 --sources <private dir> --curated aligned-curated

--curated names the curated reference (default curated; aligned-curated on a15, written as overlap-aligned-curated.json
and OVERLAP-aligned-curated.md). Where every draft has the same class-level domain-and-range pairs, the reading says
so (science, design log, 20 Oct 13:23): the drafts' mean then has almost no between-draft spread in that layer.

Reads data/sources/<graph>/ (census-t5, curated, generated-1 to 3, each through harness.sources.source_file) and
writes overlap.json (the figures, the rule, and each file's sha256 and canonical-JSON digest) and OVERLAP.md (the
table) beside them. Source structure only: no store, model, checker or test outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.sources import source_file  # noqa: E402
from harness.sources.model import OntologySource  # noqa: E402
from harness.sources.overlap import RULE, dr_pairs, overlap, to_markdown  # noqa: E402

DRAFTS = ("generated-1", "generated-2", "generated-3")


NUMBERS = {2: "two", 3: "three"}


def reading(src_dir: Path, reference: str = "curated") -> tuple[dict, str]:
    files, sources = {}, {}
    for name in ("census-t5", reference, *DRAFTS):
        path = source_file.resolve(src_dir, name)
        if path is None:
            if name in DRAFTS:
                continue
            raise SystemExit(f"{src_dir}: no {name} source")
        sources[name] = OntologySource.from_json(source_file.read_json(path))
        files[name] = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "content_sha256": source_file.content_sha256(path)}
    drafts = [n for n in DRAFTS if n in sources]
    if not drafts:
        raise SystemExit(f"{src_dir}: no generated draft")
    rows = {n: overlap(sources[n], sources["census-t5"], sources[reference]) for n in drafts}
    pairs = [dr_pairs(sources[n]) for n in drafts]
    shared = all(p == pairs[0] for p in pairs) if len(pairs) > 1 else None
    out = {"rule": RULE, "graph_dir": src_dir.name, "reference": reference, "files": files, "overlap": rows,
           "drafts_share_domain_range_set": shared}
    note = (f"The {NUMBERS.get(len(drafts), len(drafts))} drafts share one class-level domain-and-range set, so the "
            f"{NUMBERS.get(len(drafts), len(drafts))}-draft mean has almost no between-draft spread in that layer "
            "(science, design log, 20 Oct 13:23).\n\n") if shared else ""
    md = (f"# Overlap of the generated drafts ({src_dir.name})\n\n{RULE}.\n\n" + note
          + to_markdown(rows, sources["census-t5"], sources[reference]) + "\nFiles: "
          + "; ".join(f"{n} {f['file']} sha256 {f['sha256'][:16]}..." for n, f in files.items()) + ".\n")
    return out, md


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Entry 92's per-layer overlap of the generated drafts.")
    ap.add_argument("--graph", required=True)
    ap.add_argument("--sources", type=Path, help="default data/sources/<graph>")
    ap.add_argument("--curated", default="curated", help="the curated reference's source name (aligned-curated on a15)")
    args = ap.parse_args(argv)
    src_dir = args.sources or ROOT / "data" / "sources" / args.graph
    out, md = reading(src_dir, args.curated)
    suffix = "" if args.curated == "curated" else f"-{args.curated}"
    (src_dir / f"overlap{suffix}.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (src_dir / f"OVERLAP{suffix}.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
