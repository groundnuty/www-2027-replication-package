#!/usr/bin/env python3
"""Build the agent's system prompt for every prompt condition on one graph, on equal terms (E1.1, design log).

    uv run python scripts/build_prompts.py --graph rhea-142
    uv run python scripts/build_prompts.py --graph opaque-rhea-142 \\
        --leak-map ../opaque/heldout/opaque-rhea-142/map.tsv.gz

Until the generated redraw lands, a retired draw 1 can be rendered for inspection; it is marked provisional
(key ``generated-provisional``), its block is not added to the block store, and no run can select it:

    uv run python scripts/build_prompts.py --graph rhea-142 --provisional-generated \\
        data/sources/rhea-142/generated/superseded-stale-instructions/generated-1.json

Reads only data/sources/<graph>/: the sources (curated.json.gz, census-t5.json, generated-1.json: draw 1,
the fixed prompt cell), the rendering scope (used-terms.json), the shared hierarchy (shared-hierarchy.json)
and which source classes are shared-hierarchy classes (shared-terms.json.gz, read from the store by
build_sources.py, or here with --fetch-shared-terms). No store, model or key otherwise; the same inputs give
the same bytes.

Equal terms (proposal §7.3; check contract v2, rules 16 and 33):
* one renderer, harness.sources.render, with its fixed order: the source's own content through every layer,
  then its own statements on shared-hierarchy terms, then the shared hierarchy, the same text for all;
* one budget per graph, the same number for every source: the graph's ruled number (StoreLayout.render_budget,
  read through harness.sources.render_guard.ruled_budget before anything else; a graph with none is refused).
  BUDGET_RULE: B = min(entry 22's value, B_fit), entry 22's value being the largest own content among the curated
  source and the generated draws, rounded up to the next 1,000 tokens; a ruling may exceed it up to the largest
  own content of any prompted source, the census included (design log, 19 Oct 15:49: no budget may truncate a
  source). The build measures both values on the current sources with the renderer's tokenizer (tiktoken
  o200k_base) and refuses a ruling above the second; truncation by whole items, as the renderer does it, the
  census by descending support within each layer;
* one template, harness.agent.prompts: none-v1 byte for byte, then the ontology block (BLOCK_DEFINITION: the
  fixed framing then the rendered source, exactly the bytes that differ from none-v1; the budget applies to the
  rendered source, the framing being the same for every source).

Writes under data/prompts/<graph>/:
    <key>.prompt.txt   the system prompt at --row-cap and --sparql-timeout (key: none, curated, census,
                       generated, or generated-provisional)
    <key>.render.json  the renderer's manifest: block tokens, the layers that survived, the block's sha256
    manifest.json      per condition: source file and digest, block hash and tokens, surviving layers, prompt
                       hash and tokens; the budget and its rule; the inputs' digests; on an opaque graph the
                       leak check's result
    leak-check.json    on an opaque graph: the report of scripts/check_opaque_leaks.py over every prompt
    leak-scan/         on an opaque graph: a generated prompt split into what is not model-written (scanned as
                       input) and the lines the generator wrote (scanned as model output), split_model_written
and each selectable block under data/prompts/blocks/<sha256>.txt, where a run loads it. An entry is selectable
(``selectable: true``, the only kind scripts/run_pilot.py accepts) when it is not provisional and, on an opaque
graph, the input leak check passed; the manifest also names the layout and its graphs, which a run checks
against its store.

The opaque leak check (rule 34; ruling in the design log, change-log entry 20): any original name in anything not
model-written (the template, the curated and census blocks, and the renderer's own text and the shared
hierarchy inside a generated block) fails the build with exit status 1, after the files are written, to
inspect; no block is then stored and no entry is selectable. What the generator itself wrote in its block is
scanned the same way and counted per generated block in the manifest; the block is used as generated.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.agent.prompts import (  # noqa: E402
    BLOCK_DIR,
    BLOCK_PREFIX,
    ONTOLOGY_FRAMING,
    get_condition,
    ontology_block,
    sha256_text,
    store_block,
)
from harness.sources.census import DEFAULT_THRESHOLD  # noqa: E402
from harness.sources.closure import Hierarchy  # noqa: E402
from harness.sources.imported import RULE as IMPORTED_RULE  # noqa: E402
from harness.sources.imported import Imported, imported_parts, namespaces_sha256  # noqa: E402
from harness.sources.model import OntologySource  # noqa: E402
from harness.sources.render import LAYERS, PASSES, Render, SharedContext, own_content_text, render  # noqa: E402
from harness.sources import source_file  # noqa: E402
from harness.sources.render_guard import (  # noqa: E402
    RULINGS,
    UnruledRender,
    authorised_budget,
    budget_values,
    ruled_budget,
)
from harness.sources.scope import Scope, UsedTerms, compute_scope  # noqa: E402
from harness.sources.shared_terms import FILE_NAME as SHARED_TERMS_FILE  # noqa: E402
from harness.sources.shared_terms import SharedTerms, load_shared_terms, source_files  # noqa: E402
from harness.sources.tokens import count_tokens, tokenizer_info  # noqa: E402
from harness.sources.vocab import LAYOUTS, StoreLayout  # noqa: E402

SCHEMA = "prompt-manifest/1"
BUDGET_STEP = 1000
BUDGET_RULE = (
    "per graph, the graph's ruled number (StoreLayout.render_budget; a graph without one is refused), "
    "B = min(entry 22's value, B_fit), where entry 22's value is the largest own content among the curated source "
    "and the three generated draws, each rendered in full (all of its own layers, no shared-hierarchy terms: "
    "harness.sources.render.own_content_text), rounded up to the next 1,000 o200k_base tokens. The build measures "
    "entry 22's value on the current sources; a ruling below it is B_fit binding. A ruling may exceed it up to the "
    "no-truncation value, the largest own content of any prompted source (the census included) rounded up the same "
    "way, so that no source is truncated (design log, 19 Oct 15:49); a ruling above that is refused (the sources "
    "changed since the ruling). The same budget for every source on that graph, never tuned per source. "
    "Each source's own content comes first, then its statements on shared-hierarchy terms and the shared hierarchy "
    "while room remains, cut by whole items; a census is cut by descending support within each layer (science "
    "rulings in the design log, pre-registration section 5, change-log entries 20 and 22; on an opaque graph the "
    "rule is applied to its renamed curated source and its own draws)."
)
BUDGET_RULING = "change-log entry 22 (StoreLayout.render_budget)"
# The run defaults of scripts/run_pilot.py; the prompt files are written at these, the block does not depend on them.
DEFAULT_ROW_CAP = 500
DEFAULT_TIMEOUT_S = 60.0
CONDITION_ORDER = ("none", "curated", "aligned-curated", "census", "generated", "generated-2", "generated-3",
                   "generated-provisional")
# Words that say where a schema came from or name a database, counted as whole words (so not inside an
# identifier such as rh:curatedOrder). The fixed text has none (tests); a block's own content is reported,
# never repaired (rule 15).
PROVENANCE_WORDS = ("curated", "generated", "derived", "llm", "census", "rhea", "chebi", "uniprot")


@dataclass(frozen=True)
class Entry:
    key: str
    condition: str
    path: Path
    provisional: bool = False
    inspection: bool = False  # drafts 2 and 3 (StoreLayout.extra_arms): rendered and measured, never stored


@dataclass(frozen=True)
class Context:
    used: UsedTerms
    hierarchy: Hierarchy
    shared_terms: SharedTerms
    hierarchy_sha256: str
    owner_namespaces: tuple[str, ...] = ()  # rule 16 as amended (entry 101): given only where the layout asks for it


def under(path: Path, base: Path) -> str:
    """A path as the manifest writes it: relative to a directory it names (the sources or the prompts), so the
    manifest of an opaque graph carries no directory name that names the original database."""
    return os.path.relpath(Path(path).resolve(), Path(base).resolve())


def rel(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return os.path.relpath(Path(path).resolve(), ROOT)


def read_source(path: Path) -> OntologySource:
    raw = path.read_bytes()
    return OntologySource.from_json(json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw))


def load_context(src_dir: Path, owner_namespaces: tuple[str, ...] = ()) -> Context:
    hierarchy_path = src_dir / "shared-hierarchy.json"
    sh = json.loads(hierarchy_path.read_text(encoding="utf-8"))
    st_path = src_dir / SHARED_TERMS_FILE
    if not st_path.exists():
        raise SystemExit(f"{rel(st_path)} is missing: read it from the store first (--fetch-shared-terms ENDPOINT)")
    return Context(
        used=UsedTerms.from_json(json.loads((src_dir / "used-terms.json").read_text(encoding="utf-8"))),
        hierarchy=Hierarchy({c: ps for c, ps in sh["parents"].items()}),
        shared_terms=load_shared_terms(st_path),
        hierarchy_sha256=hashlib.sha256(hierarchy_path.read_bytes()).hexdigest(),
        owner_namespaces=tuple(owner_namespaces),
    )


def read_owner_namespaces(path: Path) -> tuple[str, ...]:
    """The owner's namespaces (whitespace-separated) from a private file outside the repository: rule 16 as amended
    takes them as an explicit input, and they may name a private graph's vocabulary."""
    path = Path(path).expanduser()
    if path.resolve().is_relative_to(ROOT.resolve()) or (path.stat().st_mode & 0o077):
        raise SystemExit("the owner's namespaces file must be a private file outside the repository (mode 600)")
    names = tuple(path.read_text(encoding="utf-8").split())
    if not names:
        raise SystemExit("the owner's namespaces file names no namespace")
    return names


def imported_of(src: OntologySource, ctx: Context) -> Imported | None:
    """Rule 16 as amended: the source's parts from another publisher, where the owner's namespaces are given."""
    if not ctx.owner_namespaces:
        return None
    return Imported(ctx.owner_namespaces, imported_parts(src, ctx.owner_namespaces),
                    frozenset(ctx.used.classes | ctx.used.properties))


def select_sources(src_dir: Path, provisional_generated: Path | None,
                   extra_arms: tuple[str, ...] = ()) -> tuple[list[Entry], dict[str, str]]:
    """The source of each ontology condition, and why any condition is skipped. ``extra_arms`` (the layout's,
    StoreLayout.extra_arms) adds the aligned curated source as its own condition and drafts 2 and 3 for inspection
    (rendered and measured by the fit, never stored or selectable); each is then required, and an aligned curated
    source in a graph whose layout does not list it is refused."""
    # One resolver for every builder (harness.sources.source_file; External review of ee6fd40, item 2).
    curated = source_file.resolve(src_dir, "curated")
    census = source_file.resolve(src_dir, f"census-t{DEFAULT_THRESHOLD}")
    if curated is None or census is None:
        raise SystemExit(f"{rel(src_dir)} needs curated.json.gz and census-t{DEFAULT_THRESHOLD}.json")
    entries = [Entry("curated", "curated", curated)]
    aligned = source_file.resolve(src_dir, "aligned-curated")
    if (aligned is not None) != ("aligned-curated" in extra_arms):
        raise SystemExit(f"{rel(src_dir)}: " + ("an aligned curated source is a condition only where the layout lists "
                                                "it (StoreLayout.extra_arms)" if aligned is not None else
                                                "the layout lists an aligned curated arm, and there is no "
                                                "aligned-curated source"))
    if aligned is not None:
        entries.append(Entry("aligned-curated", "aligned-curated", aligned))
    entries.append(Entry("census", "census", census))
    skipped: dict[str, str] = {}
    draw1 = source_file.resolve(src_dir, "generated-1") or src_dir / "generated-1.json"
    if draw1.exists():
        if provisional_generated is not None:
            raise SystemExit(f"{rel(draw1)} is the current draw 1; render it, not a provisional one")
        entries.append(Entry("generated", "generated", draw1))
    elif provisional_generated is not None:
        entries.append(Entry("generated-provisional", "generated", provisional_generated, provisional=True))
        skipped["generated"] = (f"no current draw 1 ({draw1.name}); {under(provisional_generated, src_dir)} is "
                                "rendered as generated-provisional, for inspection only")
    else:
        skipped["generated"] = f"no current draw 1 ({draw1.name}): the generated condition waits for the redraw"
    for key in (k for k in ("generated-2", "generated-3") if k in extra_arms):
        path = source_file.resolve(src_dir, key)
        if path is None or not draw1.exists():
            raise SystemExit(f"{rel(src_dir)}: the layout lists {key}, rendered beside the current draw 1, and "
                             f"{'there is no ' + key + ' source' if path is None else 'there is no draw 1'}")
        entries.append(Entry(key, "generated", path, inspection=True))
    return entries, skipped


def prepare(src: OntologySource, ctx: Context) -> tuple[Scope, SharedContext]:
    """Rule 33's scope and shared context, as build_sources.py computes them, from the persisted inputs."""
    terms = ctx.shared_terms.for_classes(src.classes)  # a class nobody checked stops here
    return compute_scope(src, ctx.used, terms, frozenset(ctx.hierarchy.nodes)), SharedContext(ctx.hierarchy, terms)


def render_source(src: OntologySource, ctx: Context, budget: int) -> Render:
    scope, shared = prepare(src, ctx)
    return render(src, budget, scope, shared, imported_of(src, ctx))


def own_content(src: OntologySource, ctx: Context) -> str:
    scope, shared = prepare(src, ctx)
    return own_content_text(src, scope, shared, imported_of(src, ctx))


def measured_sources(src_dir: Path, entries: list[Entry]) -> dict[str, Path]:
    """The sources the budget is measured on: the curated source and every generated draw (1 to 3) beside the
    rendered draw 1, current or provisional, or in the graph's directory when no draw 1 is rendered."""
    gen = next((e for e in entries if e.condition == "generated" and not e.inspection), None)
    base = gen.path.parent if gen is not None else src_dir
    draws = {f"generated-{i}": source_file.resolve(base, f"generated-{i}") for i in (1, 2, 3)}
    # Both curated arms count where there are two (design ruling in the design log Oct 12:54: one B for both).
    aligned = {e.key: e.path for e in entries if e.condition == "aligned-curated"}
    return {"curated": entries[0].path, **aligned, **{k: p for k, p in draws.items() if p is not None}}


def round_up_budget(tokens: int) -> int:
    return -(-tokens // BUDGET_STEP) * BUDGET_STEP


def provenance_words(text: str) -> dict[str, int]:
    low = text.lower()
    return {w: n for w in PROVENANCE_WORDS if (n := len(re.findall(rf"\b{re.escape(w)}\b", low)))}


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def write_json(path: Path, obj) -> None:
    write_text(path, json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2) + "\n")


BLOCK_DEFINITION = (
    "the ontology block is the fixed framing (a blank line, the framing sentence, a blank line) then the rendered "
    "source: exactly the bytes an ontology prompt adds to none-v1, so two conditions' prompts differ only inside it. "
    "block.sha256 is its hash, the one a run records (agent_config.ontology_block_sha256) and loads from the block "
    "store; block.tokens counts it whole. The budget applies to the rendered source alone (block.rendered_tokens, "
    "block.rendered_sha256 = the render manifest's text_sha256): the framing is the same bytes for every source."
)


def with_graphs(layout: StoreLayout, graph_file: Path | None) -> StoreLayout:
    """The layout with its graph read from ``graph_file`` where its graphs are private; a private layout without the
    file, or a file for a layout that names its graphs, is refused."""
    if not layout.private_graphs:
        if graph_file is not None:
            raise SystemExit(f"{layout.name}: the layout names its graphs; a graph file is only for a private layout")
        return layout
    if graph_file is None:
        raise SystemExit(f"{layout.name}: its graph is private; give the private graph file (--graph-file)")
    graph_file = Path(graph_file).expanduser()
    if graph_file.resolve().is_relative_to(ROOT.resolve()) or not _owner_only(graph_file):
        raise SystemExit(f"{layout.name}: the graph file must be a private file outside the repository (mode 600)")
    names = graph_file.read_text(encoding="utf-8").split()
    if len(names) != 1:
        raise SystemExit(f"{layout.name}: the graph file must name exactly one graph ({len(names)} given)")
    return dataclasses.replace(layout, data_graph=names[0], graphs=(names[0],))


def _owner_only(path: Path) -> bool:
    """Whether ``path`` grants nothing to its group or to others."""
    return (Path(path).stat().st_mode & 0o077) == 0


def _private_tree(path: Path) -> None:
    """Mode 700 for ``path`` and every directory under it, 600 for every file."""
    path = Path(path)
    if path.is_dir():
        path.chmod(0o700)
        for p in path.rglob("*"):
            p.chmod(0o700 if p.is_dir() else 0o600)


def build(graph: str, src_dir: Path, out: Path, *, row_cap: int, timeout_s: float,
          provisional_generated: Path | None = None, block_dir: Path | None = None, log=print,
          leak_map: Path | None = None, reviewed: Path | None = None, graph_file: Path | None = None,
          owner_namespaces_file: Path | None = None) -> dict:
    """Render every condition's prompt (``_build``). A layout with private graphs takes its graph from ``graph_file``,
    read here, never written in code, and its sources, prompts and blocks must be named directories outside the
    repository; everything written there is mode 600 in mode-700 directories, before and after the build."""
    layout = with_graphs(LAYOUTS[graph], graph_file)
    if layout.imports_by_origin != (owner_namespaces_file is not None):
        raise SystemExit(f"{graph}: " + ("its prompts render imported files after its own layers (rule 16 as "
                                         "amended); give the owner's namespaces file (--owner-namespaces-file)"
                                         if layout.imports_by_origin else
                                         "the owner's namespaces are an input only where the layout renders imported "
                                         "files by origin (StoreLayout.imports_by_origin)"))
    ns = read_owner_namespaces(owner_namespaces_file) if owner_namespaces_file is not None else ()
    if not layout.private_graphs:
        return _build(layout, graph, src_dir, out, row_cap=row_cap, timeout_s=timeout_s,
                      provisional_generated=provisional_generated, block_dir=block_dir, log=log, leak_map=leak_map,
                      reviewed=reviewed, owner_namespaces=ns)
    for what, d in (("sources", src_dir), ("prompts", out), ("blocks", block_dir)):
        if d is None or Path(d).resolve().is_relative_to(ROOT.resolve()):
            raise SystemExit(f"{graph}: its {what} directory must be a private directory outside the repository")
    exposed = [p for p in [Path(src_dir), *Path(src_dir).rglob("*")] if not _owner_only(p)]
    if exposed:  # the sources are read, never repaired here: an exposed source is the operator's to fix
        raise SystemExit(f"{graph}: {len(exposed)} entries of the sources directory grant access to group or others; "
                         "make it mode 700 with files 600 first")
    old = os.umask(0o077)
    try:
        for d in (out, block_dir):
            _private_tree(d)
        return _build(layout, graph, src_dir, out, row_cap=row_cap, timeout_s=timeout_s,
                      provisional_generated=provisional_generated, block_dir=block_dir, log=log, leak_map=leak_map,
                      reviewed=reviewed, owner_namespaces=ns)
    finally:
        try:
            for d in (out, block_dir):
                _private_tree(d)
        finally:
            os.umask(old)


def _build(layout: StoreLayout, graph: str, src_dir: Path, out: Path, *, row_cap: int, timeout_s: float,
           provisional_generated: Path | None, block_dir: Path | None, log, leak_map: Path | None,
           reviewed: Path | None, owner_namespaces: tuple[str, ...] = ()) -> dict:
    """Render every condition's prompt. On an opaque graph the input leak check runs before any block is stored;
    an entry is selectable by a run only when it is not provisional or for inspection and, on an opaque graph, the
    check passed."""
    try:  # before anything is read or measured: no build picks a budget itself (External reviews of 7e81a1e)
        ruled = ruled_budget(layout)
    except UnruledRender as e:
        raise SystemExit(f"{graph}: {e}") from None
    opaque = graph.startswith("opaque-")
    if opaque and leak_map is None:
        raise SystemExit(f"{graph} is opaque: the held-out map is needed for the leak check (--leak-map)")
    ctx = load_context(src_dir, owner_namespaces)
    entries, skipped = select_sources(src_dir, provisional_generated, layout.extra_arms)
    sources = {e.key: read_source(e.path) for e in entries}
    measured, files = {}, {}
    for name, path in measured_sources(src_dir, entries).items():
        src = read_source(path)
        if src.graph != layout.name:
            raise SystemExit(f"{rel(path)} is a source of {src.graph!r}, not of {layout.name!r}")
        if name != "curated" and src.name != name:
            raise SystemExit(f"{rel(path)} is {src.name}, not {name}: each draw file holds its own draw (the "
                             "generated prompt cell is draw 1)")
        measured[name] = count_tokens(own_content(src, ctx))
        files[name] = under(path, src_dir)
    largest = max(sorted(measured), key=lambda k: measured[k])
    rule_value = round_up_budget(measured[largest])  # entry 22's value on the current sources
    # Design log, 19 Oct 15:49: no budget may truncate a source, so a ruling may go up to the census's own content too.
    census_own = count_tokens(own_content(sources["census"], ctx))
    no_truncation_value = round_up_budget(max(measured[largest], census_own))
    # Only an authorised number (External review of 3831e43, item 2): the value the layout's rule gives, recomputed
    # here on the current sources, or a fit-bound number as its committed fit computation gives it, measured on the
    # sources rendered here (External review of bebd077, item 2).
    values = budget_values({**measured, "census-t5": census_own})
    assert values == {"entry-22": rule_value, "no-truncation": no_truncation_value}
    try:
        authorised_budget(layout, values, ROOT, {graph: source_file.content_digests(e.path for e in entries)},
                          {graph: namespaces_sha256(owner_namespaces) if owner_namespaces else None})
    except UnruledRender as e:
        raise SystemExit(f"{graph}: {e} (entry 22's value {rule_value}: {largest} {measured[largest]} tokens of own "
                         f"content; the no-truncation value {no_truncation_value}: census {census_own})") from None
    budget = ruled
    b_fit_binding = ruled < rule_value
    ruling = RULINGS.get(layout.name)
    truncates_own = bool(ruling is not None and ruling.truncates_own and b_fit_binding)
    own_cut: list[str] = []
    log(f"{graph}: budget {budget} tokens, ruled ({BUDGET_RULING}); entry 22's value {rule_value} (own content: "
        + ", ".join(f"{k} {n}" for k, n in measured.items()) + f"); census own content {census_own}, no-truncation "
        f"value {no_truncation_value}"
        + ("; the ruling is below entry 22's value: B_fit binding" if b_fit_binding else ""))
    none = get_condition("none")
    none_text = none.render(row_cap=row_cap, timeout_s=timeout_s)
    write_text(out / "none.prompt.txt", none_text)
    blocks: dict[str, str] = {}
    conditions = {"none": {
        "condition": "none", "id": none.id, "provisional": False,
        "prompt": {"file": "none.prompt.txt", "sha256": sha256_text(none_text),
                   "tokens": count_tokens(none_text), "template_sha256": none.template_sha256},
    }}
    for e in entries:
        src = sources[e.key]
        if src.graph != layout.name:
            raise SystemExit(f"{rel(e.path)} is a source of {src.graph!r}, not of {layout.name!r}")
        if e.condition == "generated" and src.name != ("generated-1" if not e.inspection else e.key):
            raise SystemExit(f"{rel(e.path)} is {src.name}; the generated prompt cell is draw 1")
        r = render_source(src, ctx, budget)
        if e.condition in ("curated", "aligned-curated", "generated") and any(
                x["status"] in ("partial", "none") for x in r.manifest["layers"] if x["pass"] in ("own", "imported")):
            # The rule's premise: the budget holds every measured source's own content. The truncation note could
            # in principle push one over; then the rule needs a ruling, not a quiet change. A ruling that says the
            # budget cuts own content (render_guard.Ruling.truncates_own) is that ruling: the cut is reported.
            if not truncates_own:
                raise SystemExit(f"{graph}: the own content of {e.key} does not fit the budget {budget}")
            own_cut.append(e.key)
        cond = get_condition(e.condition)
        block = ontology_block(r.text)
        prompt = cond.render(row_cap=row_cap, timeout_s=timeout_s, ontology_block=block)
        assert prompt == none_text + block  # the block is every byte the condition adds to none-v1
        block_sha = sha256_text(block)
        blocks[e.key] = block
        write_text(out / f"{e.key}.prompt.txt", prompt)
        write_json(out / f"{e.key}.render.json", r.manifest)
        layers = r.manifest["layers"]
        conditions[e.key] = {
            "condition": e.condition, "id": cond.id, "provisional": e.provisional, "selectable": False,
            **({"inspection": True} if e.inspection else {}),
            "source": {"file": under(e.path, src_dir), "file_sha256": hashlib.sha256(e.path.read_bytes()).hexdigest(),
                       "name": src.name, "kind": src.kind, "digest": src.digest()},
            "block": {
                "sha256": block_sha, "tokens": count_tokens(block), "store": None,  # set once it is selectable
                "rendered_sha256": r.manifest["text_sha256"], "rendered_tokens": r.manifest["tokens"],
                "budget_tokens": budget, "truncated": r.manifest["truncated"],
                "items_shown": r.manifest["items_shown"], "items_total": r.manifest["items_total"],
                "surviving_layers": [x["key"] for x in layers if x["status"] in ("full", "partial")],
                "layers": {x["key"]: f"{x['status']} {x['items_shown']}/{x['items_total']}" for x in layers},
                "provenance_words": provenance_words(r.text),
                "render_manifest": f"{e.key}.render.json",
            },
            "prompt": {"file": f"{e.key}.prompt.txt", "sha256": sha256_text(prompt),
                       "tokens": count_tokens(prompt), "template_sha256": cond.template_sha256},
        }
        log(f"{graph} {e.key}: rendered {r.manifest['tokens']}/{budget} tokens, block sha256 {block_sha[:16]}, layers "
            + ", ".join(f"{k}={v}" for k, v in conditions[e.key]["block"]["layers"].items() if not v.endswith("/0")))
    for reason in skipped.values():
        log(f"{graph}: {reason}")
    manifest = {
        "schema": SCHEMA,
        "layout": layout.name,  # the store layout's name: neutral on an opaque graph
        "layout_graphs": list(layout.graphs),  # a run checks its store serves these (scripts/run_pilot.py)
        "data_graph": layout.data_graph,
        "opaque": opaque,
        "store_identity": (ctx.shared_terms.store or {}).get("identity"),
        "block_definition": BLOCK_DEFINITION,
        "budget_tokens": budget,
        "budget_ruled": True,  # only a ruled number gets here (harness.sources.render_guard.ruled_budget)
        "budget_ruling": BUDGET_RULING,
        "budget_rule": BUDGET_RULE,
        "budget_measure": {"own_content_tokens": measured, "files": files, "largest": largest,
                           "census_own_content_tokens": census_own, "no_truncation_value": no_truncation_value,
                           "rounded_up_to": BUDGET_STEP, "rule_value": rule_value, "b_fit_binding": b_fit_binding},
        "tokenizer": tokenizer_info(),
        "anthropic_token_count": "not counted: messages.count_tokens needs an API key (scripts/count_prompt_tokens.py)",
        "framing": ONTOLOGY_FRAMING,
        "row_cap": row_cap,
        "sparql_timeout_s": timeout_s,
        "inputs": {
            "used_terms_digest": ctx.used.digest(),
            "shared_hierarchy_sha256": ctx.hierarchy_sha256,
            "shared_terms_checked_sha256": ctx.shared_terms.to_json()["checked_sha256"],
            "shared_terms_store": ctx.shared_terms.store,
        },
        "conditions": {k: conditions[k] for k in CONDITION_ORDER if k in conditions},
        "skipped": skipped,
        "leak_check": None,
        **({"own_content_cut": own_cut} if own_cut else {}),
        **({"imported": {"rule": IMPORTED_RULE, "owner_namespaces": list(owner_namespaces)}}
           if owner_namespaces else {}),
        # A private graph's prompts: a run reads their blocks from the private store named here (block.store).
        **({"private": True} if layout.private_graphs else {}),
    }
    if opaque:  # rule 34; ruling in the design log, change-log entry 20
        manifest["leak_check"] = leak_check(manifest, out, leak_map, reviewed)
    clean = not opaque or manifest["leak_check"]["leak"] is False
    for key, block in blocks.items():
        entry = manifest["conditions"][key]
        if entry["provisional"] or entry.get("inspection") or not clean:
            continue  # a provisional or inspection render, or any input hit on an opaque graph: nothing a run selects
        store_block(block, block_dir)
        entry["selectable"] = True
        entry["block"]["store"] = under((block_dir or BLOCK_DIR) / f"{entry['block']['sha256']}.txt", out)
    return manifest


_TITLES = {f"## {PASSES[p][1]}{layer + 1}. {LAYERS[layer][1]}": p * len(LAYERS) + layer
           for p in range(len(PASSES)) for layer in range(len(LAYERS))}
MODEL_STAGES = 2 * len(LAYERS)  # the passes "own" and "own_shared": what the source itself states
TRUNCATION_NOTE = "[Truncated at the token budget"


def split_model_written(block: str) -> tuple[str, str]:
    """A generated block's lines as (fixed, model-written). Model-written: every item of the source's own two
    passes, the generated ontology's own statements, which the model wrote. Fixed: the header, the prefix lines,
    the layer titles, the truncation note, and the shared hierarchy (the same text for every source). The prefix
    lines the model-written lines use are repeated with them, so prefixed names resolve in both scans. Items are
    one line each (labels and comments are written as JSON strings)."""
    fixed, model, prefixes = [], [], []
    stage = None
    for line in block.splitlines():
        if line in _TITLES:
            stage = _TITLES[line]
            fixed.append(line)
        elif not line or line.startswith(TRUNCATION_NOTE) or stage is None or stage >= MODEL_STAGES:
            fixed.append(line)
            if stage is None and line.startswith("PREFIX "):
                prefixes.append(line)
        else:
            model.append(line)
    used = [p for p in prefixes if any(re.search(rf"(?<![\w-]){re.escape(p.split()[1])}", m) for m in model)]
    return "\n".join(fixed) + "\n", "\n".join(used + model) + "\n"


def leak_check(manifest: dict, out: Path, map_path: Path, reviewed: Path | None) -> dict:
    """The independent scan (scripts/check_opaque_leaks.py, which imports nothing from the builders). Every prompt
    is an input, whose hits fail the build, except the lines a generator wrote in a generated block: those are
    scanned as model output and counted (ruling in the design log, change-log entry 20)."""
    inputs, outputs = {}, {}
    for key, c in manifest["conditions"].items():
        prompt = (out / c["prompt"]["file"]).read_text(encoding="utf-8")
        if c["condition"] != "generated":
            inputs[key] = rel(out / c["prompt"]["file"])
            continue
        template = get_condition("none").render(row_cap=manifest["row_cap"],
                                                timeout_s=manifest["sparql_timeout_s"]) + BLOCK_PREFIX
        assert prompt.startswith(template)
        fixed, model = split_model_written(prompt[len(template):])
        write_text(out / "leak-scan" / f"{key}.fixed.txt", template + fixed)
        write_text(out / "leak-scan" / f"{key}.model-written.txt", model)
        inputs[key] = rel(out / "leak-scan" / f"{key}.fixed.txt")
        outputs[key] = rel(out / "leak-scan" / f"{key}.model-written.txt")
    report = out / "leak-check.json"
    cmd = [sys.executable, "scripts/check_opaque_leaks.py", "--map", rel(map_path), "--report", rel(report),
           "--input", *inputs.values(), "--output", *outputs.values()]
    if reviewed is not None:
        cmd += ["--reviewed", rel(reviewed)]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)  # exit 1 is a finding
    if proc.returncode not in (0, 1):
        raise SystemExit(f"the leak check failed to run: {proc.stderr.strip()[-2000:]}")
    found = json.loads(report.read_text(encoding="utf-8"))
    return {
        "script": "scripts/check_opaque_leaks.py",
        "map": "the held-out map, at a private path (not recorded here)",
        "reviewed_file": reviewed.name if reviewed is not None else None,  # beside the sources
        "report": report.name,  # the check's own report; it names the files and the map's path
        "scanned": {k: {"input": under(ROOT / f, out), "model_written": under(ROOT / outputs[k], out) if k in outputs
                        else None} for k, f in inputs.items()},
        "hits": {k: found["inputs"][f]["counts"] for k, f in inputs.items()},
        "model_written_findings": {k: found["model_outputs"][f]["counts"] for k, f in outputs.items()},
        "reviewed_hits": {k: sorted(found["inputs"][f]["reviewed"]) for k, f in inputs.items()},
        "leak": bool(found["leak"]),
    }


def fetch_shared_terms(graph: str, src_dir: Path, endpoint: str, extra: Path | None, timeout_s: float) -> SharedTerms:
    from build_sources import store_identity  # scripts/; reads the store's stats only

    from harness.sources.shared_terms import read_shared_terms, write_shared_terms
    from harness.sparql import SparqlClient

    layout = LAYOUTS[graph]
    candidates: set[str] = set()
    for path in source_files(src_dir) + ([extra] if extra is not None else []):
        candidates |= set(read_source(path).classes)
    st = read_shared_terms(SparqlClient(endpoint, timeout_s=timeout_s), layout, candidates,
                           store_identity(endpoint, layout.name))
    write_shared_terms(src_dir / SHARED_TERMS_FILE, st)
    return st


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--graph", required=True, choices=sorted(LAYOUTS))
    ap.add_argument("--sources", type=Path, help="default data/sources/<graph>")
    ap.add_argument("--out", type=Path, help="default data/prompts/<graph>")
    ap.add_argument("--block-dir", type=Path, default=None, help="default data/prompts/blocks")
    ap.add_argument("--row-cap", type=int, default=DEFAULT_ROW_CAP)
    ap.add_argument("--sparql-timeout", type=float, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--provisional-generated", type=Path, help="a retired draw 1, rendered as generated-provisional")
    ap.add_argument("--leak-map", type=Path, help="the held-out map (default ../opaque/heldout/<graph>/map.tsv.gz)")
    ap.add_argument("--fetch-shared-terms", metavar="ENDPOINT",
                    help="first read the shared-term status of every source class from this store")
    ap.add_argument("--timeout", type=float, default=290.0, help="client time-out of the store queries")
    ap.add_argument("--graph-file", type=Path, help="the private file naming the graph of a private layout")
    ap.add_argument("--owner-namespaces-file", type=Path,
                    help="the private file naming the owner's namespaces (rule 16 as amended; layouts that ask)")
    args = ap.parse_args(argv)
    if LAYOUTS[args.graph].private_graphs and None in (args.sources, args.out, args.block_dir):
        raise SystemExit(f"{args.graph}: a private graph's sources, prompts and blocks stay in private directories; "
                         "give --sources, --out and --block-dir")
    if LAYOUTS[args.graph].private_graphs and args.fetch_shared_terms:
        raise SystemExit(f"{args.graph}: a private graph's render context is written from its files "
                         "(scripts/render_context_from_files.py), never fetched here")
    src_dir = args.sources or ROOT / "data" / "sources" / args.graph
    out = args.out or ROOT / "data" / "prompts" / args.graph
    if args.fetch_shared_terms:
        st = fetch_shared_terms(args.graph, src_dir, args.fetch_shared_terms, args.provisional_generated, args.timeout)
        print(f"{args.graph}: {len(st.checked)} source classes checked, {len(st.shared)} shared", file=sys.stderr)
    map_path = reviewed = None
    if args.graph.startswith("opaque-"):  # rule 34: every opaque prompt is scanned
        map_path = args.leak_map or ROOT.parent / "opaque" / "heldout" / args.graph / "map.tsv.gz"
        if not map_path.exists():
            raise SystemExit(f"opaque graph: the held-out map is needed for the leak check (--leak-map); "
                             f"{rel(map_path)} does not exist")
        reviewed = src_dir / "leak-review.json"
        reviewed = reviewed if reviewed.exists() else None
    manifest = build(args.graph, src_dir, out, row_cap=args.row_cap,
                     timeout_s=args.sparql_timeout, provisional_generated=args.provisional_generated,
                     block_dir=args.block_dir, log=lambda m: print(m, file=sys.stderr), leak_map=map_path,
                     reviewed=reviewed, graph_file=args.graph_file, owner_namespaces_file=args.owner_namespaces_file)
    status = 0
    if manifest["leak_check"] is not None:
        lc = manifest["leak_check"]
        bad = {k: v for k, v in lc["hits"].items() if v}
        print(f"{args.graph}: leak check " + (f"FOUND INPUT HITS in {bad}" if bad else "found 0 input hits")
              + f"; model-written findings in generated blocks: {lc['model_written_findings']}", file=sys.stderr)
        status = 1 if manifest["leak_check"]["leak"] else 0
    if LAYOUTS[args.graph].private_graphs:  # the manifest names the private graph: written private, as the prompts
        old = os.umask(0o077)
        try:
            write_json(out / "manifest.json", manifest)
            _private_tree(out)
        finally:
            os.umask(old)
    else:
        write_json(out / "manifest.json", manifest)
    return status


if __name__ == "__main__":
    sys.exit(main())
