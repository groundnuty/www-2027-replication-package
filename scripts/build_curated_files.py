"""Build a curated source from a published ontology directory, read whole (rule 15, as published), privately.

    python -I scripts/build_curated_files.py --name <graph name> --dir <ontology directory> \\
        --turtle <file> ... --rdfxml <file> ... --not-rdf <file> ... \\
        --expect-digest <sha256> --provenance <private json> --out <private directory>

Every file of the directory must be named exactly once, with its syntax (or as not RDF: hashed in the directory digest,
not loaded). The directory digest (``LC_ALL=C sha256sum * | LC_ALL=C sort | sha256sum``) must equal --expect-digest,
which must equal the provenance record's ontology digest. The source is read by ``build_curated_from_files``, the
curated reader every curated source goes through (change-log entry 94's class rule included), each file with its own
document base (--base: the directory's published location, else each file's own location), and written with the
provenance record's ontology fields to <out>/curated.json, mode 600.

Every loaded file stays in the source ("as published"; design ruling in the design log Oct 05:24), and the private
build record <out>/curated-build.json gives each file's share of it: the classes,
subclass axioms, disjoint pairs and domain and range declarations of the source read from that file alone, of the
source read without it, and the whole source's minus the latter (what the file adds). Only counts are printed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.sources.curated import build_curated_from_files, directory_digest  # noqa: E402


def write_private(path: Path, obj) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.chmod(path, 0o600)


SHARE_COUNTS = ("classes", "subclass_axioms", "disjoint_pairs", "domain_range_declarations")


def file_shares(name: str, files, base, whole: dict) -> list[dict]:
    """Per loaded file: the source's counts from that file alone and without it, and what it adds to the whole."""
    out = []
    for i, (path, fmt) in enumerate(files):
        alone = build_curated_from_files(name, [(path, fmt)], base).provenance["counts"]
        rest = [f for j, f in enumerate(files) if j != i]
        without = build_curated_from_files(name, rest, base).provenance["counts"] if rest else dict.fromkeys(
            SHARE_COUNTS, 0)
        out.append({"file": Path(path).name, "format": fmt,
                    "alone": {k: alone[k] for k in SHARE_COUNTS},
                    "without": {k: without[k] for k in SHARE_COUNTS},
                    "share": {k: whole[k] - without[k] for k in SHARE_COUNTS}})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--turtle", nargs="*", default=[])
    ap.add_argument("--rdfxml", nargs="*", default=[])
    ap.add_argument("--not-rdf", nargs="*", default=[])
    ap.add_argument("--expect-digest", required=True)
    ap.add_argument("--provenance", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--base", default=None, help="the directory's published location (each file's base is it + name)")
    args = ap.parse_args()
    os.umask(0o077)

    named = [*args.turtle, *args.rdfxml, *args.not_rdf]
    present = sorted(p.name for p in args.dir.iterdir() if not p.name.startswith("."))
    if sorted(named) != present or len(set(named)) != len(named):
        raise SystemExit(f"every file of the directory must be named exactly once ({len(present)} files, "
                         f"{len(named)} named, {len(set(named))} distinct)")
    digest = directory_digest(args.dir)
    if digest != args.expect_digest:
        raise SystemExit("the directory digest is not the expected one; nothing is built")
    prov = json.loads(args.provenance.read_text())
    if prov.get("ontology_digest_sha256") != digest or not str(prov.get("ontology_version", "")).endswith(digest):
        raise SystemExit("the provenance record names another ontology digest or version; nothing is built")

    files = [(args.dir / f, "turtle") for f in sorted(args.turtle)] + [(args.dir / f, "xml") for f in sorted(args.rdfxml)]
    src = build_curated_from_files(args.name, files, args.base)
    src.provenance["ontology"] = {
        "commit": prov.get("ontology_commit"), "digest_sha256": digest, "version": prov.get("ontology_version"),
        "kg_snapshot": prov.get("kg_snapshot"), "source": prov.get("ontology_source"),
        "files_hashed": len(present), "files_loaded": len(files), "not_rdf": sorted(args.not_rdf),
        "digest_recipe": "LC_ALL=C sha256sum * | LC_ALL=C sort | sha256sum, over every file of the directory",
        "condition": "unaligned curated: the ontology as published, without any mapping to the data's IRIs",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    os.chmod(args.out, 0o700)
    write_private(args.out / "curated.json", src.to_json())
    counts = dict(src.provenance["counts"])
    counts.update(classes_by_namespace=len(counts.pop("classes_by_namespace")),
                  properties_by_namespace=len(counts.pop("properties_by_namespace")),
                  described=len(src.described or {}),
                  declared_classes="all listed" if src.declared_classes is None else len(src.declared_classes),
                  unsupported_total=sum(src.unsupported.values()), files_loaded=len(files),
                  graph_triples=src.provenance["read"]["graph_triples"])
    summary = {"digest_ok": True, "digest": digest, "source_sha256": src.digest(), "counts": counts}
    shares = file_shares(args.name, files, args.base, src.provenance["counts"])
    write_private(args.out / "curated-build.json", {**summary, "file_shares": shares})
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
