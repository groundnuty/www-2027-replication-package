"""Build the ontology sources for one graph, their renders and their profiles (E0.3, E1b.1).

    uv run python scripts/build_sources.py --endpoint http://localhost:7019/ \\
        --graph rhea-142 --out data/sources/rhea-142/

Writes, under --out:
    used-terms.json          the terms the data uses (the rendering/profiling scope, shared by all sources)
    shipped-terms.json       the shipped-vocabulary terms in any source's scope (rendered in pass 2)
    curated.json             the curated source (full; the checkers use the full closure)
    census-t<T>.json         the census at each threshold (check contract v2, rule 20)
    census-t<T>.json.gz      the census at each --offline-thresholds threshold (default 2 and 20; design log): the same
                             recipe and the same count, for model-free offline replay only; never rendered, profiled
                             or drawn (a bundle is drawable only from census-t5, ``generated.is_drawable``)
    generated/input/         the generator's input bundle and its sha256, always counted with the free
                             token-counting endpoint (rule 33's fallback applied): a bundle is drawable only
                             with a passing count; an earlier, different bundle is moved to generated/superseded/
    generated/<retired>/     draws made from an earlier bundle, with their input, generated sources, renders and
                             profiles (--retire-as; default superseded-<request sha256 prefix>): never current
    generated-<i>.json       a generated source, for each successful draw of the current bundle
    shared-terms.json.gz     which source classes are shared-hierarchy classes, as read from the store, so the
                             renders rebuild offline (scripts/build_prompts.py)
    renders/<name>.txt       each source rendered at the graph's budget, with <name>.manifest.json
    profiles.json, PROFILES.md

The budget is the graph's ruled one (``StoreLayout.render_budget``; change-log entry 22: 6,000 o200k tokens on
Rhea and on opaque Rhea, the largest own content among curated and the generated draws, rounded up to the next
1,000; design log, entries 22, 56 and 60 and 19 Oct 15:49: 50,000 on UniProt), the same for every source; a graph with
no ruled number needs --budget. Rule 16 renders a source's own terms through every layer before the shipped
vocabulary (ChEBI), so the budget is spent first on what distinguishes the sources; a census is cut by descending
support within each layer (entry 22, ``harness.sources.render``); the manifests say what survived.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.sources import generated as gen  # noqa: E402
from harness.sources.census import (  # noqa: E402
    DEFAULT_THRESHOLD,
    MEMBERSHIP_PREDICATES,
    build_censuses,
    mixed_positions,
)
from harness.sources.curated import (  # noqa: E402
    DOC_ADDITIONS,
    PUBLISHED,
    as_published,
    build_curated,
    with_documented_terms,
)
from harness.sources import hierarchy_adapter as ha  # noqa: E402
from harness.sources import source_file  # noqa: E402
from harness.sources.membership import fetch_census_input, undeclared_terms  # noqa: E402
from harness.sources.model import OntologySource  # noqa: E402
from harness.sources.profile import overlap_with_curated, profile, to_markdown  # noqa: E402
from harness.sources.render import SharedContext, own_content_text, render  # noqa: E402
from harness.sources.render_guard import UnruledRender, authorised_budget, budget_values  # noqa: E402
from harness.sources.scope import SCOPE_RULE, UsedTerms, compute_scope  # noqa: E402
from harness.sources.shared_terms import FILE_NAME as SHARED_TERMS_FILE  # noqa: E402
from harness.sources.shared_terms import SharedTerms, write_shared_terms  # noqa: E402
from harness.sources.tokens import count_tokens, tokenizer_info  # noqa: E402
from harness.sources.vocab import LAYOUTS, PREFIXES, StoreLayout, membership_label, ref  # noqa: E402
from harness.sparql import SparqlClient  # noqa: E402


def write_json(path: Path, obj, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    else:
        text = json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2)
    path.write_text(text + "\n", encoding="utf-8")


def write_json_gz(path: Path, obj) -> None:
    text = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
    with open(path, "wb") as fh, gzip.GzipFile(fileobj=fh, mode="wb", mtime=0) as gz:
        gz.write(text.encode("utf-8"))


def log(msg: str) -> None:
    import time

    print(time.strftime("%H:%M:%S ") + msg, file=sys.stderr, flush=True)


def store_identity(endpoint: str, layout_name: str) -> dict:
    """The serving store's identity from QLever's ``?cmd=stats`` (decision c): its index name (``<store>@<sha256
    of index.sha256>`` where the build set one), or the fact that the server reports none, with the index's
    triple counts and build hash as further evidence of what was read. The store label before ``@`` is written as
    the layout's name: an opaque store's label names the original database, which no opaque file may hold; the
    sha256 after ``@`` is the index's own and identifies it."""
    import httpx

    url = endpoint.rstrip("/") + "/?cmd=stats"
    try:
        stats = httpx.get(url, timeout=30).json()
    except (httpx.HTTPError, ValueError) as e:
        return {"stats_url": url, "identity": None, "note": f"the server's stats could not be read: {e}"}
    name = stats.get("name-index")
    named = bool(name) and name != "no index name specified"
    label, _, sha = (name or "").rpartition("@") if named else ("", "", "")
    identity = f"{layout_name}@{sha}" if named and sha else (name if named else None)
    return {
        "stats_url": url,
        "identity": identity,
        "note": ("the index name the server reports, its store label written as the layout's name"
                 + ("" if label == layout_name else " (the server's label differs)") if named
                 else f"the server reports no index name ({name!r})"),
        "git_hash_index": stats.get("git-hash-index"),
        "num_triples_normal": stats.get("num-triples-normal"),
        "num_predicates_normal": stats.get("num-predicates-normal"),
        "num_subjects_normal": stats.get("num-subjects-normal"),
    }


def vocabulary_manifest(client, layout: StoreLayout, store: dict | None = None) -> list[dict]:
    """Rule 16: each source records the vocabulary versions and hashes it was built against."""
    inputs: dict[str, list[str]] = {}
    tsv = ROOT / "stores" / layout.name / "inputs.tsv"
    if tsv.exists():
        for line in tsv.read_text().splitlines():
            if line and not line.startswith("#"):
                path, graph = line.split("\t")[:2]
                inputs.setdefault(graph, []).append(path)
    expected = {}
    exp = ROOT / "stores" / layout.name / "expected_counts.json"
    if exp.exists():
        expected = json.loads(exp.read_text())
    out = []
    for g in layout.graphs:
        n = client.query(f"SELECT (COUNT(*) AS ?n) WHERE {{ GRAPH {ref(g)} {{ ?s ?p ?o }} }}").rows[0][0]
        onto = client.query(PREFIXES + (
            f"SELECT ?o ?v WHERE {{ GRAPH {ref(g)} {{ ?o a owl:Ontology OPTIONAL {{ ?o owl:versionIRI|owl:versionInfo ?v }} }} }}"
        )).rows
        role = ("data" if g == layout.data_graph else "shipped" if g in layout.shipped_graphs
                else "excluded (source under comparison)" if g in layout.excluded_graphs else "other")
        files = sorted(inputs.get(g, []))
        out.append({
            "graph": g,
            "role": role,
            "triples": int(n.lexical),
            "expected_triples": expected.get(g),
            "input_files": files,
            # A hash of the input file paths only; content hashes of the release files live on the cluster.
            "input_paths_sha256": hashlib.sha256("\n".join(files).encode()).hexdigest() if files else None,
            "store": store,
            "ontology": sorted(([r[0].value, getattr(r[1], "lexical", getattr(r[1], "value", None))] for r in onto),
                               key=lambda x: (x[0], x[1] or "")),
        })
    return out


def current_decontaminated(out: Path, natural: dict | None) -> list[OntologySource]:
    """The decontaminated sources (generated-dc-<i>.json), each checked against the current decontaminated bundle
    (out/generated-dc/input): made from its exact request, with the map it records, and that bundle renamed from the
    current natural bundle. Anything else is refused, as a natural draw from another bundle is (External reviews of
    cf5c000): rerun scripts/build_decontaminated.py bundle, which retires stale draws."""
    paths = sorted(out.glob("generated-dc-*.json"))
    if not paths:
        return []
    dc_dir = out / "generated-dc"
    if not (dc_dir / "input" / "bundle.json").exists():
        raise RuntimeError(f"{[p.name for p in paths]} have no decontaminated bundle at {dc_dir}/input; retire them")
    dc_bundle = gen.load_bundle(dc_dir)
    dc_info = dc_bundle.get("decontamination") or {}
    if natural is None or dc_info.get("natural_request_sha256") != natural["request_sha256"]:
        raise RuntimeError(f"the decontaminated bundle was renamed from natural request {dc_info.get('natural_request_sha256')}, "
                           f"not the current {natural and natural['request_sha256']}; rebuild it")
    out_sources = []
    for path in paths:
        src = OntologySource.from_json(json.loads(path.read_text()))
        made = src.provenance.get("decontamination") or {}
        draw = src.provenance.get("draw") or {}
        if (made.get("request_sha256") != dc_bundle["request_sha256"] or draw.get("request_sha256") != dc_bundle["request_sha256"]
                or made.get("map_sha256") != dc_info.get("map_sha256") or draw.get("map_sha256") != dc_info.get("map_sha256")):
            raise RuntimeError(f"{path.name} was made from request {draw.get('request_sha256')} with map "
                               f"{draw.get('map_sha256')}, not the current {dc_bundle['request_sha256']} with map "
                               f"{dc_info.get('map_sha256')}; retire it before building")
        out_sources.append(src)
    return out_sources


PROMPTED = ("curated", "census-t5", "generated-1", "generated-2", "generated-3")


def prompted_sources(out: Path, built: list[OntologySource]) -> list[OntologySource]:
    """Every prompted source of the graph (curated, census-t5, generated-1 to 3): the one built in this invocation,
    else its file already in ``out`` (External review of 8a224a3, item 4: a source on disk can invalidate the budget
    as build_prompts measures it)."""
    by_name = {s.name: s for s in built if s.name in PROMPTED}
    for name in PROMPTED:
        # The file prompt assembly reads too (harness.sources.source_file; External review of ee6fd40, item 2): refused
        # when <name>.json and <name>.json.gz hold different sources, a source built here included.
        path = source_file.resolve(out, name)
        if name not in by_name and path is not None:
            by_name[name] = OntologySource.from_json(source_file.read_json(path))
    return [by_name[n] for n in PROMPTED if n in by_name]


def own_tokens(src: OntologySource, used: UsedTerms, shared_terms, shared_h) -> int:
    """A source's own content in o200k tokens, measured as scripts/build_prompts.py measures it (own_content_text)."""
    scope = compute_scope(src, used, shared_terms, frozenset(shared_h.nodes))
    return count_tokens(own_content_text(src, scope, SharedContext(shared_h, shared_terms)))


def measure_budget_values(sources: list[OntologySource], used: UsedTerms, shared_terms, shared_h) -> dict[str, int]:
    """Entry 22's value and the no-truncation value (render_guard.budget_values) of the given prompted sources."""
    return budget_values({src.name: own_tokens(src, used, shared_terms, shared_h) for src in sources
                          if src.name in PROMPTED})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the ontology sources, renders and profiles of one graph.")
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--graph", required=True, choices=sorted(LAYOUTS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--thresholds", type=int, nargs="+", default=[DEFAULT_THRESHOLD])
    ap.add_argument("--offline-thresholds", type=int, nargs="*", default=[2, 20],
                    help="censuses written as census-t<T>.json.gz for offline replay only (never sources, never drawn)")
    ap.add_argument("--budget", type=int, default=None,
                    help="render budget in o200k tokens, the same for every source (default: the graph's ruled budget)")
    ap.add_argument("--no-generated-input", action="store_true", help="skip the generator's input bundle")
    ap.add_argument("--retire-as", default=None,
                    help="directory under generated/ for draws made from an earlier bundle (default superseded-<sha16>)")
    ap.add_argument("--retire-reason", default="the generator's input changed; these draws were made from an earlier "
                                               "bundle and are never current sources")
    ap.add_argument("--key-file", type=Path, default=Path.home() / ".config" / "www-2027" / "anthropic_api_key")
    ap.add_argument("--timeout", type=float, default=290.0)
    ap.add_argument("--sources", nargs="+", default=["curated", "census", "generated"],
                    choices=["curated", "census", "generated"], help="which sources to build")
    args = ap.parse_args(argv)
    if set(args.offline_thresholds) & ({DEFAULT_THRESHOLD} | set(args.thresholds)):
        raise SystemExit(f"--offline-thresholds {args.offline_thresholds} overlaps the source thresholds {args.thresholds}")
    builds_input = "generated" in args.sources and "census" in args.sources and not args.no_generated_input
    if builds_input and DEFAULT_THRESHOLD not in args.thresholds:
        raise SystemExit(f"rule 33: the generator's input is the census at threshold {DEFAULT_THRESHOLD} "
                         f"(census-t{DEFAULT_THRESHOLD}), which --thresholds {args.thresholds} does not build; nothing is "
                         "drawn from another threshold (add it, or pass --no-generated-input)")

    layout = LAYOUTS[args.graph]
    if layout.private_graphs:
        raise SystemExit(f"{args.graph}: its graph is private and its sources are built from files "
                         "(scripts/build_curated_files.py, build_census_onegraph.py, generated_from_files.py); "
                         "build_sources never reads its store")
    if args.budget is None:
        if layout.render_budget is None:
            raise SystemExit(f"{args.graph} has no ruled render budget (StoreLayout.render_budget is unset); name one "
                             "with --budget")
        args.budget = layout.render_budget
    # Ruled only at the graph's ruled number, and only once its rule recomputes that number on the sources built here
    # (harness.sources.render_guard.authorised_budget, before rendering); any other render is marked unruled, and
    # prompt assembly refuses it.
    at_ruled_number = layout.render_budget is not None and args.budget == layout.render_budget
    client = SparqlClient(args.endpoint, timeout_s=args.timeout)
    out: Path = args.out

    log("census input ...")
    cin = fetch_census_input(client, layout)
    store = store_identity(args.endpoint, layout.name)
    log(f"store: {store['identity'] or store['note']}")
    vocab = vocabulary_manifest(client, layout, store)
    log("listing and shared hierarchy (rules 33 and 35) ...")
    listing = ha.fetch_listing(client, layout, cin.keep_subject("y"))
    used_shared = ha.fetch_used_shared(client, layout, cin.keep_subject("s"))
    shared_h = ha.fetch_shared_parents(client, layout, used_shared | {c for c in listing.listed})
    used = UsedTerms(frozenset(used_shared), ha.fetch_used_predicates(client, layout))
    write_json(out / "used-terms.json", used.to_json())

    sources: list[OntologySource] = []
    curated = None
    if "curated" in args.sources:
        log("curated ...")
        if layout.name in PUBLISHED:  # rule 15: refused unless the store graph is the published file
            published = as_published(client, layout, ROOT)
        curated = build_curated(client, layout)
        curated.provenance["vocabularies"] = vocab
        if layout.name in PUBLISHED:
            curated.provenance["published"] = published
        write_json(out / "curated.json", curated.to_json(), compact=True)
        sources.append(curated)
        if layout.name in DOC_ADDITIONS:  # offline checker variant (design log ruling); not rendered or profiled
            variant = with_documented_terms(curated, layout.name)
            text = json.dumps(variant.to_json(), sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
            with open(out / "curated+doc.json.gz", "wb") as fh, gzip.GzipFile(fileobj=fh, mode="wb", mtime=0) as gz:
                gz.write(text.encode("utf-8"))

    # The census, its sample and the count are made in memory first, and the bundle they make stale is retired
    # before the census is written, so no failure part-way leaves an earlier bundle drawable against a new census
    # (External review of the sources snapshot, item 2); the draw command also checks the census (census_current).
    censuses, offline = [], []
    if "census" in args.sources:
        thresholds = [*args.thresholds, *args.offline_thresholds]
        log(f"census t={thresholds} (one count) ...")
        built = build_censuses(client, layout, thresholds, cin=cin, listing=listing, shared=shared_h, log=log)
        for t in args.thresholds:
            census = built[t]
            census.provenance["vocabularies"] = vocab
            censuses.append(census)
            sources.append(census)
        for t in args.offline_thresholds:  # design log: offline replay only, so neither a source nor a generator input
            census = built[t]
            census.provenance["vocabularies"] = vocab
            census.provenance["use"] = ("model-free offline replay only (design log): never rendered into a prompt, never "
                                        f"a generator input; only census-t{DEFAULT_THRESHOLD} is drawable")
            offline.append(census)
    main_census = next((c for c in censuses if c.support_threshold == DEFAULT_THRESHOLD), None)

    gen_dir = out / "generated"
    bundle, tokens = None, None
    if "generated" in args.sources and censuses and not args.no_generated_input:
        log("generator input bundle ...")
        if main_census is None:  # rule 33: never another threshold's census (External review of cf5c000, item 1)
            raise SystemExit(f"rule 33: no census-t{DEFAULT_THRESHOLD} was built; the generator's input is never drawn "
                             "from another threshold")
        sample = gen.sample_instances(client, layout, main_census)
        # Always counted on the exact request, with rule 33's fallback (External review of b0c3920, item 6).
        api = gen.anthropic_client(args.key_file)
        bundle, fit = gen.fit_bundle(main_census, sample, lambda b: gen.count_input(b, api)["input_tokens"])
        log(f"  rule 33 input fit: {fit}")
        tokens = count_tokens(bundle["system"]) + count_tokens(bundle["user"])

    if main_census is not None:  # with --no-generated-input too: a bundle of another census never stays drawable
        dest = gen.retire_stale_bundle(gen_dir, out, main_census, bundle and bundle["request_sha256"],
                                       args.retire_as)
        if dest is not None:
            log(f"  the bundle built from another census or replaced by this build retired to {dest}")
    for census in censuses:
        write_json(out / f"{census.name}.json", census.to_json(), compact=True)
    for census in offline:
        write_json_gz(out / f"{census.name}.json.gz", census.to_json())
    if bundle is not None:
        stale = gen.draws_request_sha(gen_dir) - {bundle["request_sha256"]} if gen_dir.exists() else set()
        if stale:
            name = args.retire_as or f"superseded-{sorted(stale)[0][:16]}"
            dest = gen.retire_draws(gen_dir, out, name, args.retire_reason)
            log(f"  draws made from {sorted(s_[:16] for s_ in stale)} retired to {dest}")
        gen.write_bundle(bundle, gen_dir, tokens)
        log(f"  request sha256 {bundle['request_sha256']}, {tokens:,} o200k tokens, "
            f"{bundle['sample_spec']['n_instances']} instances, status {bundle['status']}")
    if "generated" in args.sources:
        if bundle is None and (gen_dir / "input" / "bundle.json").exists():
            bundle = gen.load_bundle(gen_dir)
        for i in range(1, gen.N_DRAWS + 1):
            src = gen.load_draw(gen_dir, i, layout.name, membership_label(layout))
            if src is None:
                continue
            drawn_from = src.provenance.get("draw", {}).get("request_sha256")
            if bundle is None or drawn_from != bundle["request_sha256"]:  # External review of b0c3920, item 7
                raise RuntimeError(f"draw {i} was made from bundle {drawn_from}, not the current "
                                   f"{bundle and bundle['request_sha256']}; retire it (--retire-as) before building")
            src.provenance["vocabularies"] = vocab
            write_json(out / f"{src.name}.json", src.to_json(), compact=True)
            sources.append(src)

    if "generated" in args.sources:  # decontaminated drafts, mapped back by scripts/build_decontaminated.py
        sources += current_decontaminated(out, bundle)

    try:  # measured for the budget's ruling, built here or already on disk
        prompted, conflict = prompted_sources(out, sources), None
    except ValueError as e:
        prompted, conflict = [], str(e)
    log("shared-hierarchy terms of the sources ...")
    candidates = frozenset(t for s_ in [*sources, *prompted] for t in s_.classes)
    shared_terms = ha.fetch_shared_terms(client, layout, candidates)
    write_shared_terms(out / SHARED_TERMS_FILE, SharedTerms(candidates, frozenset(shared_terms), store))
    context = SharedContext(shared_h, shared_terms)
    write_json(out / "shared-hierarchy.json", {
        "definition": ha.DEFINITION,
        "hierarchy_graphs": list(ha.hierarchy_graphs(layout)),
        "rule": "rule 33: the shared-hierarchy classes the data uses and their ancestors, with their published "
                "subclass edges in the hierarchy graphs; rendered the same for every source",
        "classes": len(shared_h.nodes),
        "parents": {c: list(ps) for c, ps in sorted(shared_h.parents.items())},
    }, compact=True)
    undeclared = None
    if curated is not None:
        log("terms the data uses that the curated source does not declare ...")
        undeclared = undeclared_terms(client, layout, set(curated.classes), set(curated.properties))

    budget_ruled = False
    if at_ruled_number and conflict is not None:
        log(f"  renders marked unruled: {conflict}")
    elif at_ruled_number:
        try:
            # A fit-bound number must have been measured on the prompted sources on disk, the files prompt assembly
            # reads (External review of bebd077, item 2).
            on_disk = source_file.content_digests(p for n in PROMPTED if (p := source_file.resolve(out, n)) is not None)
            authorised_budget(layout, measure_budget_values(prompted, used, shared_terms, shared_h), ROOT,
                              {args.graph: on_disk})
            budget_ruled = True
        except UnruledRender as e:
            log(f"  renders marked unruled: {e}")
    budget_rule = "change-log entry 22 (StoreLayout.render_budget)" if budget_ruled else None
    profiles, render_summary = [], []
    for src in sources:
        scope = compute_scope(src, used, shared_terms, frozenset(shared_h.nodes))
        r = render(src, args.budget, scope, context)
        (out / "renders").mkdir(parents=True, exist_ok=True)
        (out / "renders" / f"{src.name}.txt").write_text(r.text, encoding="utf-8")
        write_json(out / "renders" / f"{src.name}.manifest.json",
                   {**r.manifest, "budget_ruled": budget_ruled, "budget_rule": budget_rule})
        render_summary.append({"source": src.name, "tokens": r.manifest["tokens"],
                               "layers": {x["key"]: x["status"] for x in r.manifest["layers"]}})
        prof = profile(src, scope, used.properties)
        if curated is not None and src.kind == "generated":
            prof["overlap_with_curated"] = overlap_with_curated(curated, src, shared_terms)
        profiles.append(prof)
        log(f"  {src.name}: render {r.manifest['tokens']} tokens, layers "
            + ", ".join(f"{x['key']}={x['status']}({x['items_shown']}/{x['items_total']})"
                        for x in r.manifest["layers"] if x["items_total"]))

    doc = {
        "graph": layout.name,
        "store": store,
        "used_terms_digest": used.digest(),
        "scope_rule": SCOPE_RULE,
        "membership_reading": cin.reading(),
        "definition": ha.DEFINITION,
        "render_budget": args.budget,
        "render_budget_ruled": budget_ruled,
        "tokenizer": tokenizer_info(),
        "vocabularies": vocab,
        "profiles": profiles,
        "renders": render_summary,
        "undeclared_in_curated": undeclared,
    }
    write_json(out / "profiles.json", doc)
    (out / "PROFILES.md").write_text(profiles_markdown(
        doc, layout, censuses[0] if censuses else None, curated.provenance if curated is not None else None, used),
        encoding="utf-8")
    log("done")
    return 0


# The census listing's keys in the order harness.sources.census writes them. Its note prints them in this order whether
# the census is the one just built or read back from its key-sorted file, so the committed profiles regenerate offline.
LISTING_ORDER = ("listed", "primary_targets", "secondary_targets", "data_nodes", "secondary_not_named",
                 "entity_class_targets", "data_node_test", "rule")
# The example a graph's census-scope remark gives (design ruling, design log, 20 Oct 10:34): (predicate, wording). Given
# only while that predicate is among the used predicates the census does not record, and never on an opaque graph.
SCHEMA_PREDICATE_EXAMPLES = {
    "rhea-142": ("http://rdf.rhea-db.org/coefficient", "rh:coefficient on the rh:containsN properties"),
}


def census_remarks(census: OntologySource, used: UsedTerms, layout: StoreLayout) -> list[str]:
    """The profile's remarks on the census's predicates. The census-scope one (design log, 20 Oct 10:34), where the census
    records fewer predicates than the data uses: the predicates whose only triples are the schema's own are outside its
    data scope. The mixed-position count (change-log entry 97, ``census.mixed_positions``), naming the predicates except
    on an opaque graph."""
    used_predicates = set(used.properties) - set(MEMBERSHIP_PREDICATES)
    outside = used_predicates - set(census.properties)
    out = []
    if outside:
        example = SCHEMA_PREDICATE_EXAMPLES.get(layout.name)
        named = f" (for example {example[1]})" if example and example[0] in outside and not layout.opaque else ""
        out.append(f"Census data scope (design log, 20 Oct 10:34): the profile counts {len(used_predicates)} used predicates "
                   f"and {census.name} records {len(census.properties)}; schema-level predicates{named} are outside the "
                   "census's data scope.")
    mixed = mixed_positions(census)
    line = (f"Mixed census positions (change-log entry 97): {len(mixed)} of the {len(census.properties)} predicates "
            f"{census.name} records have literal and IRI objects both at or over the threshold "
            f"({census.support_threshold}), so the census derives no range there (kind unknown) and an IRI constant at "
            "such a position is not_applicable for domain/range, judged only through joins.")
    if mixed and not layout.opaque:
        line += " Predicates: " + ", ".join(f"`{p}`" for p in mixed) + "."
    return [*out, line]


def profile_notes(doc: dict, layout: StoreLayout, census: OntologySource | None, curated: dict | None) -> list[str]:
    """The profile's notes, from profiles.json's content ``doc``, the census and the curated source's provenance."""
    notes = [f"Membership: {doc['membership_reading']}", f"Definition: {doc['definition']}"]
    if not ha.hierarchy_graphs(layout):  # design log, 19 Oct 23:02: reported in the profile
        notes.append("Shared hierarchy: empty. No graph read other than the data graph and the compared source "
                     f"({', '.join(layout.excluded_graphs) or 'none'}) holds subclass edges (rules 16 and 35), so "
                     "membership is read from the data graph alone and the compared source's own subclass axioms "
                     "reach no other source.")
    if census is not None:
        census_input = census.provenance["input"]
        listing = census.provenance["listing"]
        listing = {**{k: listing[k] for k in LISTING_ORDER if k in listing}, **listing}
        notes += [
            f"Census input (check contract v2, rule 20): graphs {census_input['census_graphs']}; excluded graphs "
            f"{census_input['excluded_graphs']}; the schema's own triples left out: "
            f"{census_input['excluded_schema_triples_total']} triples of {len(census_input['excluded_schema_subjects'])} "
            "schema subjects in the data graph.",
            f"Census listing (rule 33): {listing}",
            f"Census domain and range: {census.provenance['domain_range_rule']}",
        ]
    notes.append(f"Scope: {doc['scope_rule']}")
    if curated is not None:
        notes.append(
            f"Curated classes by namespace: {curated['counts']['classes_by_namespace']}; "
            f"properties by namespace: {curated['counts']['properties_by_namespace']}; "
            f"excluded: {curated['excluded']}.")
    notes.append(f"Renders at {doc['render_budget']} tokens ({doc['tokenizer']['encoding']}): "
                 + "; ".join(f"{s['source']} {s['tokens']} tokens" for s in doc["renders"]))
    return notes


def profiles_markdown(doc: dict, layout: StoreLayout, census: OntologySource | None, curated: dict | None,
                      used: UsedTerms) -> str:
    """PROFILES.md from profiles.json's content ``doc``, the census (the first one built), the curated source's
    provenance and the terms the data uses: the build writes it, and ``profiles_markdown_offline`` rebuilds it from
    the committed files."""
    remarks = census_remarks(census, used, layout) if census is not None else None
    return to_markdown(doc["profiles"], profile_notes(doc, layout, census, curated), doc["undeclared_in_curated"],
                       remarks)


def profiles_markdown_offline(out: Path, layout: StoreLayout) -> str:
    """``profiles_markdown`` from a graph's committed files: profiles.json, used-terms.json, census-t5 and the curated
    source, where present. No store is read."""
    census_file = source_file.resolve(out, f"census-t{DEFAULT_THRESHOLD}")
    curated_file = source_file.resolve(out, "curated")
    census = OntologySource.from_json(source_file.read_json(census_file)) if census_file is not None else None
    curated = source_file.read_json(curated_file)["provenance"] if curated_file is not None else None
    used = UsedTerms.from_json(json.loads((out / "used-terms.json").read_text(encoding="utf-8")))
    doc = json.loads((out / "profiles.json").read_text(encoding="utf-8"))
    return profiles_markdown(doc, layout, census, curated, used)


if __name__ == "__main__":
    sys.exit(main())
