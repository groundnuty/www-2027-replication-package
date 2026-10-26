"""The render context of a graph whose census read no hierarchy graph, written from its source files with no store query.

    python -I scripts/render_context_from_files.py --sources <private sources directory>

scripts/build_prompts.py and scripts/fit_budget.py read three context files beside the sources, which
scripts/build_sources.py otherwise reads from the store (rule 33): the shared hierarchy, which source classes are
shared-hierarchy classes, and the terms the data uses. Where the graph's census records that it read no hierarchy graph
(``provenance["hierarchy_graphs"] == []``), there is no shared hierarchy, so all three follow from the files:

* shared-hierarchy.json: no classes and no edges;
* shared-terms.json.gz: every class of every source in the directory checked, none shared (a shared-hierarchy class is
  the subject or object of a subclass edge in a hierarchy graph, and there is none);
* used-terms.json: no used shared classes, and the census's predicates as the used properties (rule 33's scope keeps
  every property of a source whatever this list holds, so it changes no render; its digest is recorded).

A census that read a hierarchy graph, or does not say, is refused. Files are mode 600 in a mode-700 directory, and only
counts and digests are printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from generated_from_files import _write_private, private_writes

from harness.sources import hierarchy_adapter as ha
from harness.sources import source_file
from harness.sources.model import OntologySource
from harness.sources.scope import UsedTerms
from harness.sources.shared_terms import FILE_NAME as SHARED_TERMS_FILE
from harness.sources.shared_terms import SharedTerms, write_shared_terms

SOURCES = ("curated", "aligned-curated", "census-t5", "generated-1", "generated-2", "generated-3")
STORE_NOTE = {"identity": None, "note": "no store read: the census read no hierarchy graph, so no class is shared"}


class ContextRefused(RuntimeError):
    pass


def write_context(src_dir: Path) -> dict:
    census_path = source_file.resolve(src_dir, "census-t5")
    if census_path is None:
        raise ContextRefused("no census-t5 source in the directory")
    census = OntologySource.from_json(source_file.read_json(census_path))
    if census.provenance.get("hierarchy_graphs") != []:
        raise ContextRefused("the census does not record that it read no hierarchy graph")
    found = {n: source_file.resolve(src_dir, n) for n in SOURCES}
    sources = {n: OntologySource.from_json(source_file.read_json(p)) for n, p in found.items() if p is not None}
    classes = frozenset(c for s in sources.values() for c in s.classes)
    used = UsedTerms(frozenset(), frozenset(census.properties))
    hierarchy = {"definition": ha.DEFINITION, "hierarchy_graphs": [],
                 "rule": "rule 33: the shared-hierarchy classes the data uses and their ancestors, with their published "
                         "subclass edges in the hierarchy graphs; rendered the same for every source; here none, as "
                         "the census read no hierarchy graph",
                 "classes": 0, "parents": {}}
    hierarchy_text = json.dumps(hierarchy, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
    with private_writes(src_dir):
        _write_private(src_dir / "shared-hierarchy.json", hierarchy_text.encode())
        write_shared_terms(src_dir / SHARED_TERMS_FILE, SharedTerms(classes, frozenset(), STORE_NOTE))
        _write_private(src_dir / "used-terms.json",
                       (json.dumps(used.to_json(), sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode())
    return {"classes_checked": len(classes), "shared": 0, "hierarchy_classes": 0,
            "used_properties": len(used.properties), "sources": sorted(sources), "used_terms_digest": used.digest(),
            "shared_hierarchy_sha256": hashlib.sha256(hierarchy_text.encode()).hexdigest()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sources", type=Path, required=True)
    args = ap.parse_args(argv)
    umask = os.umask(0o077)
    try:
        print(json.dumps(write_context(args.sources), sort_keys=True))
        return 0
    except ContextRefused as exc:  # its message names no term
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    finally:
        os.umask(umask)


if __name__ == "__main__":
    sys.exit(main())
