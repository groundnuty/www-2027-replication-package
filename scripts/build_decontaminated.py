"""Decontaminated generations (design log, criterion 4): the natural generator input, renamed, then drawn.

    uv run python scripts/build_decontaminated.py bundle --graph uniprot-sp-2026_03 --seed <integer> \\
        --natural data/sources/uniprot-sp-2026_03/generated --out data/sources/uniprot-sp-2026_03/generated-dc \\
        --map heldout/decontaminated-uniprot-sp-2026_03/map.json.gz
    uv run python -m harness.sources.generated --out data/sources/uniprot-sp-2026_03/generated-dc
    uv run python scripts/build_decontaminated.py collect --graph uniprot-sp-2026_03 \\
        --out data/sources/uniprot-sp-2026_03/generated-dc --map heldout/decontaminated-uniprot-sp-2026_03/map.json.gz \\
        --sources data/sources/uniprot-sp-2026_03

``bundle`` reads the census and the instance sample back from the natural bundle (``census_block`` is
lossless, the sample is copied as it is), so the decontaminated input is exactly the natural one renamed.
Rule 33 applies identically: the renamed input is always counted on the exact request and, above 800,000
tokens, the sample is reduced uniformly down to one instance per class; without a passing count the bundle
is not drawable. The renaming is ``decontaminate`` with ``--literals`` (default ``strings``: every literal
that is not a W3C number, date or boolean masked, as proposal §7.3 says; ``names`` renames only labels and
comments) and the required ``--seed``, recorded in the bundle's provenance and the held-out map's header. The
map goes to ``heldout/``; it is never shown to a generator. The renamed bundle is checked by the builder's own
``decontaminate.verify``; ``scripts/check_opaque_leaks.py`` is the independent scan of the written files.

A map is a pure function of the input and the seed. A map already at ``--map`` is reused only when rebuilding it
from the current input gives the same map; otherwise ``bundle`` refuses, or with ``--rebuild-map`` moves the old
map aside (``<map>.superseded-<sha16>``) and writes the new one. The renamed input is verified against the
current input's names, not only the map's. Draws made from another request or with another map are retired to
``<out>/superseded-<sha16>/`` with the sources built from them (as natural draws are) before the new bundle is
written.

``collect`` maps each successful draw back through the inverse map, keeps invented terms as they are and
reports them, parses the result as a generated source (``generated-dc-<i>.json``) and writes the map-back
report beside each draw. It refuses a draw made from another request or with another map than the current
bundle's, and records both hashes in the source's provenance, where ``build_sources`` checks them again.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.sources import decontaminate as dc  # noqa: E402
from harness.sources import generated as gen  # noqa: E402
from harness.sources.tokens import count_tokens  # noqa: E402
from harness.sources.vocab import LAYOUTS, membership_label  # noqa: E402


def read_natural(natural: Path, graph: str):
    """The natural bundle's census and sample, read through ``gen.load_bundle``, which checks its prompt files against
    its recorded request hash: a partly replaced bundle never passes for the request it records (external reviewer 61 review of
    the sources snapshot, item 3)."""
    bundle = gen.load_bundle(natural)
    user = bundle["user"]
    start = user.index("<census_and_sample>\n") + len("<census_and_sample>\n")
    payload = json.loads(user[start:user.index("\n</census_and_sample>")])
    census = gen.census_from_block(payload["census"], "census-t5", graph)
    census.provenance = {k: payload["census"][k] for k in ("membership", "support", "domain_range_rule") if k in payload["census"]}
    census.provenance["membership_reading"] = payload["census"]["membership"]
    census.provenance["untyped_sentinel"] = dc._expand(payload["census"]["untyped"]) if payload["census"].get("untyped") else None
    sample = gen.InstanceSample(payload["sample"]["spec"], payload["sample"]["instances"])
    natural_record = {k: v for k, v in bundle.items() if k not in ("system", "user")}
    return census, sample, natural_record


def load_map(path: Path) -> dc.OpaqueMap:
    return dc.OpaqueMap.from_json(json.loads(gzip.decompress(path.read_bytes())))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stale_draws(out: Path, request_sha256: str, map_sha256: str) -> list[str]:
    """The draws in ``out`` made from another request or with another map (a batch in flight counts by its request)."""
    stale = []
    for p in sorted(out.glob("draw-*.json")):
        rec = json.loads(p.read_text())
        if rec.get("request_sha256") != request_sha256 or rec.get("map_sha256") != map_sha256:
            stale.append(p.name)
    if (out / "batches.json").exists() and json.loads((out / "batches.json").read_text()).get("request_sha256") != request_sha256:
        stale.append("batches.json")
    return stale


def cmd_bundle(args) -> int:
    census, sample, natural = read_natural(args.natural, LAYOUTS[args.graph].name)
    header = {"natural_request_sha256": natural["request_sha256"], "graph": args.graph,
              "builder": "scripts/build_decontaminated.py"}
    m = dc.build_map(census, sample, literals=args.literals, seed=args.seed)
    new_map = None  # the bytes of a map to write, once everything it needs has been made
    if args.map.exists():
        # A map is a function of the input, the seed and the policy: the file is reused only if it holds exactly the
        # map of the current input under the current policy (its header aside), so no name of the input is left out
        # of it (External review 61, item 1) and its recorded policy is the one applied.
        on_disk = json.loads(gzip.decompress(args.map.read_bytes()))
        on_disk.pop("header", None)
        current = m.to_json()
        current.pop("header")
        if on_disk == current:
            m = load_map(args.map)
        elif not args.rebuild_map:
            raise SystemExit(f"{args.map} is not the map of the current input under the current policy (rebuilt from "
                             "the input with the given seed it differs); pass --rebuild-map to replace it, which retires "
                             "every draw made with it")
        else:
            new_map = m.to_bytes(header)
    else:
        new_map = m.to_bytes(header)
    map_sha = hashlib.sha256(new_map).hexdigest() if new_map is not None else sha256_file(args.map)
    rc, rs = dc.rename_census(census, m), dc.rename_sample(sample, m)
    # Always counted on the exact request, with rule 33's fallback (External review of b0c3920, item 6). Nothing is
    # written before the bundle is made, and the stale bundle and draws are retired before the map or the bundle is
    # written (External review of the sources snapshot, item 2), so a failure part-way leaves nothing drawable that was
    # renamed with another map than the one at --map.
    api = gen.anthropic_client(args.key_file)
    bundle, fit = gen.fit_bundle(rc, rs, lambda b: gen.count_input(b, api)["input_tokens"])
    report = dc.verify(m, bundle["user"], dc.input_originals(census, sample, m.literals))
    stale = stale_draws(args.out, bundle["request_sha256"], map_sha) if args.out.exists() else []
    if stale:  # External reviews of cf5c000: a stale draw never becomes a current source
        sources = args.sources or args.out.parent
        dest = gen.retire_draws(args.out, sources, f"superseded-{bundle['request_sha256'][:16]}",
                                "the decontaminated input or its map changed; these draws were made from an earlier "
                                "bundle or map and are never current sources", source_prefix="generated-dc")
        print(f"draws {stale} retired to {dest}", file=sys.stderr)
    old = args.out / "input" / "bundle.json"
    if old.exists():
        prev = json.loads(old.read_text())
        if (prev.get("request_sha256") != bundle["request_sha256"]
                or (prev.get("decontamination") or {}).get("map_sha256") != map_sha):
            gen.supersede_input(args.out, bundle["request_sha256"], "the decontaminated input or its map changed; this "
                                                                    "bundle is never drawn")
    if new_map is not None:
        if args.map.exists():
            args.map.rename(args.map.with_name(f"{args.map.name}.superseded-{sha256_file(args.map)[:16]}"))
        args.map.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(args.map, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)  # held out: readable by its owner only
        with open(fd, "wb") as fh:
            fh.write(new_map)
    bundle["decontamination"] = {
        "natural_request_sha256": natural["request_sha256"],
        # The map's path names the original graph, which no file of the renamed bundle may hold: its hash is recorded.
        "map": "the held-out map given with --map", "map_sha256": map_sha,
        "seed": m.seed, "map_seed_recorded_in": "this provenance and the held-out map's header",
        "policy": {"literals": m.literals, "base": m.base},
        "verify": {k: v for k, v in report.items() if k != "leaks"}, "leaks": report["leaks"],
    }
    if not report["ok"]:
        bundle["status"] = "refused: the renamed input leaks original names"
    tokens = count_tokens(bundle["system"]) + count_tokens(bundle["user"])
    gen.write_bundle(bundle, args.out, tokens)
    print(json.dumps({"request_sha256": bundle["request_sha256"], "status": bundle["status"], "input_fit": fit,
                      "o200k_tokens": tokens, "verify": report}, indent=1))
    return 0 if report["ok"] else 1


def cmd_collect(args) -> int:
    m = load_map(args.map)
    bundle = gen.load_bundle(args.out)
    map_sha = sha256_file(args.map)
    out = []
    for i in range(1, gen.N_DRAWS + 1):
        rec_path, txt_path = args.out / f"draw-{i}.json", args.out / f"draw-{i}.txt"
        if not rec_path.exists():
            continue
        record = json.loads(rec_path.read_text())
        if record.get("status") != "ok":
            continue
        if (why := dc.stale_draw(record, bundle, map_sha)) is not None:
            raise SystemExit(f"draw {i} is not current: {why}; rerun bundle, which retires it")
        turtle, where = gen.extract_turtle(txt_path.read_text(encoding="utf-8"))
        back, report = dc.map_back_turtle(turtle, m)
        (args.out / f"draw-{i}.mapped-back.ttl").write_text(back, encoding="utf-8")
        (args.out / f"draw-{i}.mapback.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        src = gen.parse_generated(back, f"generated-dc-{i}", LAYOUTS[args.graph].name,
                                  {"draw": record, "turtle_from": where, "mapped_back": report,
                                   "decontamination": {"request_sha256": bundle["request_sha256"], "map_sha256": map_sha,
                                                       "natural_request_sha256":
                                                           bundle["decontamination"]["natural_request_sha256"]}},
                                  membership_label(LAYOUTS[args.graph]))
        (args.sources / f"generated-dc-{i}.json").write_text(
            json.dumps(src.to_json(), sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        out.append({"draw": i, "invented": len(report["invented_opaque_terms"]),
                    "original_namespace_terms": len(report["original_namespace_terms"]),
                    "classes": len(src.classes), "properties": len(src.properties)})
    print(json.dumps(out, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bundle")
    b.add_argument("--graph", required=True, choices=sorted(LAYOUTS))
    b.add_argument("--natural", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--map", type=Path, required=True)
    b.add_argument("--literals", choices=["strings", "names"], default="strings")
    b.add_argument("--seed", type=int, required=True, help="the map's seed (required; recorded in the bundle and the map)")
    b.add_argument("--rebuild-map", action="store_true",
                   help="replace a map at --map that is not the current input's (the old one is moved aside)")
    b.add_argument("--sources", type=Path, default=None,
                   help="the graph's sources directory, for retiring stale draws (default: the parent of --out)")
    b.add_argument("--key-file", type=Path, default=Path.home() / ".config" / "www-2027" / "anthropic_api_key")
    c = sub.add_parser("collect")
    c.add_argument("--graph", required=True, choices=sorted(LAYOUTS))
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--map", type=Path, required=True)
    c.add_argument("--sources", type=Path, required=True)
    args = ap.parse_args(argv)
    return cmd_bundle(args) if args.cmd == "bundle" else cmd_collect(args)


if __name__ == "__main__":
    sys.exit(main())
