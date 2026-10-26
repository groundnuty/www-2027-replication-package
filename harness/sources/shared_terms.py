"""Which source classes are shared-hierarchy classes, kept on disk so renders rebuild offline.

The scope and the renderer (rule 33) need, for every class a source declares, whether it is a class of the
shared hierarchy: a subject or object of a subclass edge in the hierarchy graphs
(``hierarchy_adapter.fetch_shared_terms``). That is a property of the store. The shared hierarchy's scoped
nodes do not answer it: they also hold the listed classes of the data graph, and they leave out the shared
classes the data never uses, which the scope must drop. So the status is read from the store once and
written here: the classes that were checked and which of them are shared. A class that was never checked (a
redrawn source's new IRI) stops the build; it never falls through to "own".
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from harness.sources import source_file
from harness.sources.census import DEFAULT_THRESHOLD
from harness.sources.hierarchy_adapter import fetch_shared_terms

FILE_NAME = "shared-terms.json.gz"
DEFINITION = ("a source class is a shared-hierarchy class when it is the subject or object of an rdfs:subClassOf "
              "edge in the hierarchy graphs (rule 35; harness.sources.hierarchy_adapter.fetch_shared_terms)")


def _digest(terms: Iterable[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(terms), separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class SharedTerms:
    checked: frozenset[str]
    shared: frozenset[str]
    store: dict | None  # the store it was read from (build_sources.store_identity)

    def __post_init__(self):
        if not self.shared <= self.checked:
            raise ValueError(f"{len(self.shared - self.checked)} shared terms were not checked")

    def for_classes(self, classes: Iterable[str]) -> frozenset[str]:
        classes = set(classes)
        missing = sorted(classes - self.checked)
        if missing:
            raise ValueError(f"{len(missing)} class(es) were not checked against the shared hierarchy "
                             f"(first: {missing[0]}); re-read the shared terms from the store")
        return frozenset(classes & self.shared)

    def to_json(self) -> dict:
        return {
            "definition": DEFINITION,
            "store": self.store,
            "checked": sorted(self.checked),
            "checked_sha256": _digest(self.checked),
            "not_shared": sorted(self.checked - self.shared),  # the short list: most checked classes are shared
            "shared_count": len(self.shared),
        }

    @staticmethod
    def from_json(obj: dict) -> SharedTerms:
        checked = frozenset(obj["checked"])
        if _digest(checked) != obj.get("checked_sha256"):
            raise ValueError("shared-terms digest does not match its checked classes")
        out = SharedTerms(checked, checked - frozenset(obj["not_shared"]), obj.get("store"))
        if len(out.shared) != obj.get("shared_count"):
            raise ValueError("shared-terms count does not match its content")
        return out


def write_shared_terms(path: Path, st: SharedTerms) -> None:
    text = json.dumps(st.to_json(), sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh, gzip.GzipFile(fileobj=fh, mode="wb", mtime=0) as gz:
        gz.write(text.encode("utf-8"))


def load_shared_terms(path: Path) -> SharedTerms:
    return SharedTerms.from_json(json.loads(gzip.decompress(Path(path).read_bytes())))


def read_shared_terms(client, layout, candidates: Iterable[str], store: dict | None) -> SharedTerms:
    """Check ``candidates`` against the store's hierarchy graphs (bounded VALUES queries, one at a time)."""
    checked = frozenset(candidates)
    return SharedTerms(checked, fetch_shared_terms(client, layout, checked), store)


def source_files(src_dir: Path) -> list[Path]:
    """Every source file of a graph's directory whose classes a render may need: the curated source, the
    censuses, and every generated draw (current, decontaminated or retired), each as the builders resolve it
    (harness.sources.source_file: the .json.gz form too; External review of bebd077, item 3). The censuses are the
    prompted census-t5 in either form and any other source threshold's census-t<T>.json; an offline census, written
    census-t<T>.json.gz only (scripts/build_sources.py), is never rendered and stays out."""
    src_dir = Path(src_dir)
    names = [(src_dir, "curated"), (src_dir, f"census-t{DEFAULT_THRESHOLD}")]
    names += sorted({(src_dir, source_file.name_of(p)) for p in src_dir.glob("census-t*.json")} - set(names))
    draws = {(p.parent, source_file.name_of(p)) for suffix in (".json", ".json.gz")
             for p in src_dir.rglob(f"generated-*{suffix}") if "renders" not in p.relative_to(src_dir).parts}
    names += sorted(draws)
    return [p for d, name in names if (p := source_file.resolve(d, name)) is not None]
