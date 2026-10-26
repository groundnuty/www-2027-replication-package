#!/usr/bin/env python3
"""Replay the frozen fault bank through every bank source's checks (dress rehearsal 2, B2; prereg section 3 "Bank
metrics"): the checked items of one split (untouched controls, rewrites, answer-changing and answer-neutral mutations;
never a descriptive or rejected draw), each read as a gold query by each source (matrix.check_bank_item), in resumable
chunks, recording every store response so that the matrix replays offline. The ledger exporter reads its matrix
(scripts/export_checker_ledger.py --matrix ...), and the analysis loads its rows as the registered bank arms.

    uv run python scripts/bank_replay.py --manifest data/banks/uniprot-sp-2026_03/manifest.json \\
        --frozen-config data/analysis/prereg-config.json --split dev [--questions IDS] \\
        --sources-dir data/sources/uniprot-sp-2026_03 [--sources curated,census-t5,generated-dc-1,generated-dc-2,generated-dc-3] \\
        [--source-file NAME=PATH ...] [--union NAME=A,B] \\
        --hierarchy data/checks/uniprot-sp-2026_03/shared-hierarchy-v4.json.gz \\
        (--endpoint http://node-1.cluster-a.example:7020/ [--store-json STORE.json] | --offline OUT/store-log.jsonl.gz) \\
        [--shard K/N] [--chunk 20] [--memory-gb 10] [--note TEXT] --out OUT
    uv run python scripts/bank_replay.py ... --merge-only --out OUT    # merge once every shard is done

--manifest must be the frozen config's bank_manifest_sha256 and its items (items.jsonl or items.jsonl.gz) the bytes it
froze. --questions declares a subset of the split's questions (a JSON array of ids, or one per line; a question with no
checked item of the split is refused): a dress rehearsal's slice. --shard K/N takes the chunks whose index is K modulo
N, so N processes share one OUT; a run stopped at any point resumes with the same command (done chunks are kept,
the rest checked), and OUT's plan.json must be the same plan (inputs, code, store, shard count N) or the run
refuses. The last shard
to finish merges (matrix.jsonl.gz, store-log.jsonl.gz, run.json: the same bytes however the run was sharded). Never
run --graph-counts-like scans: the store is read only through the checks' own bounded queries and ?cmd=stats.

The test split's bank verdicts are test outcomes (design ruling, design log, 20 Oct): --split test is refused unless
--confirmatory, the running code is the freeze tag's commit (--freeze-repo, a git clone of this repository whose
origin holds --freeze-tag), by commit and by the tree digest recomputed from git's objects, and the pre-registration is
frozen (--prereg-repo, --prereg-commit: the test grid's gate); this is checked before anything is read or any store
is contacted. On a host with no git clone (SiteA: its project space is group-readable and the repository's objects
hold the held-out blobs) --freeze-manifest replaces those four flags: a manifest made and verified on the VM
(scripts/bank_freeze_manifest.py), against which the deploy's commit, tree digest and config and the frozen
pre-registration's bytes are checked (bankrun.require_freeze_manifest). A confirmatory or test-split run checks every
item: --questions is refused there.

A live run records the store's identity (its ?cmd=stats index name and counts) in the plan, checks it again before
each chunk is written, and stops when the store did not answer a check's query (an outage would otherwise read as
"unknown" verdicts); resume it once the store is back.

--memory-gb caps the process's address space (RLIMIT_AS), so a run that outgrows it fails instead of starving the host.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.agent.replay_check import code_commit  # noqa: E402
from harness.analysis.inputs import InputError, read_question_ids  # noqa: E402
from harness.checkloop.arms import hierarchy_sha256  # noqa: E402
from harness.checkloop.bankrun import (  # noqa: E402
    DEFAULT_CHUNK,
    DEFAULT_SOURCES,
    BankReplayError,
    all_done,
    checked_items,
    log_progress,
    merge,
    open_run,
    parse_shard,
    plan,
    read_store_logs,
    require_freeze_deploy,
    require_freeze_manifest,
    run_chunks,
    store_fingerprint,
    store_stats,
    utc_now,
)
from harness.checkloop.ledger import LedgerError, frozen_bank_sha256  # noqa: E402
from harness.checkloop.matrix import Arms  # noqa: E402
from harness.checkloop.store_log import CHECKER_ROW_CAP, CHECKER_TIMEOUT_S  # noqa: E402
from harness.checks.compiler import CheckCompiler  # noqa: E402
from harness.checks.hierarchy import SharedHierarchy  # noqa: E402
from harness.sources.model import OntologySource  # noqa: E402
from harness.sparql import SparqlClient  # noqa: E402


CANARY = "SELECT ?s WHERE { ?s ?p ?o } LIMIT 1"  # answered from the first row of an index permutation


def frozen_preregistration(repo: Path | None, commit: str | None) -> dict:
    """The test grid's gate (scripts/build_dev_grid.py frozen_preregistration): ``commit`` of the science repository
    holds 'Status: frozen' in paper/preregistration.md and is on its origin's main as fetched now."""
    if repo is None or not commit:
        raise BankReplayError("a confirmatory test-split run needs --prereg-repo and --prereg-commit: the frozen "
                              "pre-registration opens the test split, as it does for the test grid")
    spec = importlib.util.spec_from_file_location("build_dev_grid_gate", ROOT / "scripts" / "build_dev_grid.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_dev_grid_gate"] = module
    spec.loader.exec_module(module)
    try:
        return module.frozen_preregistration(repo, commit)
    except module.Refused as exc:
        raise BankReplayError(str(exc)) from None


def source_path(directory: Path, name: str, files: dict[str, Path]) -> Path:
    if name in files:
        return files[name]
    for suffix in (".json.gz", ".json"):
        path = directory / f"{name}{suffix}"
        if path.exists():
            return path
    raise BankReplayError(f"no source {name!r} in {directory}")


def load_source(path: Path) -> OntologySource:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return OntologySource.from_json(json.load(f))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def say(text: str) -> None:
    print(f"{utc_now()} {text}", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--manifest", type=Path, required=True, help="the frozen bank's manifest.json")
    p.add_argument("--frozen-config", type=Path, required=True, help="the prereg config naming bank_manifest_sha256")
    p.add_argument("--split", required=True, choices=("dev", "test"))
    p.add_argument("--questions", type=Path, help="a declared subset of the split's questions (JSON array or lines)")
    p.add_argument("--sources-dir", type=Path)
    p.add_argument("--sources", default=",".join(DEFAULT_SOURCES))
    p.add_argument("--source-file", action="append", default=[], metavar="NAME=PATH")
    p.add_argument("--union", action="append", default=[], metavar="NAME=SRC1,SRC2", help="an arm read from sources")
    p.add_argument("--hierarchy", type=Path)
    p.add_argument("--endpoint", help="the store the checks read")
    p.add_argument("--timeout-s", type=float, default=CHECKER_TIMEOUT_S)
    p.add_argument("--store-json", type=Path, help="the store's description (its identity), recorded in the plan")
    p.add_argument("--offline", type=Path, help="a merged store-log.jsonl.gz to serve instead of the store")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--chunk", type=int, default=DEFAULT_CHUNK, help="items per chunk (the unit of resumption)")
    p.add_argument("--memory-gb", type=float, default=0, help="cap the address space (RLIMIT_AS); 0: no cap")
    p.add_argument("--confirmatory", action="store_true", help="the confirmatory run (required for --split test)")
    p.add_argument("--freeze-repo", type=Path, help="a git clone of this repository whose origin holds the freeze tag")
    p.add_argument("--freeze-tag", help="the freeze tag's name")
    p.add_argument("--prereg-repo", type=Path, help="the science repository whose origin's main holds the frozen "
                                                     "pre-registration (split test)")
    p.add_argument("--prereg-commit", help="the science commit whose paper/preregistration.md says 'Status: frozen'")
    p.add_argument("--freeze-manifest", type=Path,
                   help="instead of the four git flags above: the freeze manifest made on the VM "
                        "(scripts/bank_freeze_manifest.py), for a host with no git clone (split test)")
    p.add_argument("--freeze-manifest-sha256",
                   help="the manifest's sha256 as the VM printed it (with --freeze-manifest: the trust anchor)")
    p.add_argument("--note", default="", help="a line for the progress log (e.g. the deploy and the store's job)")
    p.add_argument("--merge-only", action="store_true", help="merge a finished OUT; check nothing")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        # first, before anything is read or any store contacted: the test split is a confirmatory freeze deploy's only
        if args.questions and (args.confirmatory or args.split == "test"):
            raise BankReplayError("--questions declares a subset, never a confirmatory or test-split run: those check "
                                  "every item of the split")
        git_flags = [f for f, v in (("--freeze-repo", args.freeze_repo), ("--freeze-tag", args.freeze_tag),
                                    ("--prereg-repo", args.prereg_repo), ("--prereg-commit", args.prereg_commit)) if v]
        if args.freeze_manifest is not None and git_flags:
            raise BankReplayError(f"either --freeze-manifest or the git gate's flags ({', '.join(git_flags)}), not both")
        if (args.freeze_manifest is not None or args.freeze_manifest_sha256) and args.split != "test":
            raise BankReplayError("--freeze-manifest opens the test split only; a dev run takes none")
        prereg = None  # the frozen pre-registration as checked, for the progress log
        if args.freeze_manifest is not None:  # the same checks against the manifest the VM verified and wrote
            confirmatory = require_freeze_manifest(args.split, confirmatory=args.confirmatory,
                                                   manifest=args.freeze_manifest, frozen_config=args.frozen_config,
                                                   manifest_sha256=args.freeze_manifest_sha256)
            if confirmatory is not None:
                prereg = {**confirmatory["preregistration"],
                          "freeze_manifest_sha256": confirmatory["freeze_manifest_sha256"]}
        else:
            confirmatory = require_freeze_deploy(args.split, confirmatory=args.confirmatory,
                                                 freeze_repo=args.freeze_repo, freeze_tag=args.freeze_tag)
            if confirmatory is not None:  # and the pre-registration is frozen, as the test grid's gate reads it
                prereg = frozen_preregistration(args.prereg_repo, args.prereg_commit)
                confirmatory = {**confirmatory,
                                "preregistration": {k: prereg[k] for k in ("commit", "preregistration_sha256")}}
        if args.merge_only:
            the_plan = json.loads((args.out / "plan.json").read_text(encoding="utf-8"))
            if the_plan.get("split") != args.split:
                raise BankReplayError(f"{args.out} is a run of split {the_plan.get('split')!r}, not {args.split!r}")
            if confirmatory is not None and confirmatory != the_plan.get("confirmatory"):
                raise BankReplayError(f"{args.out} was planned under another freeze record than this merge's gate gives "
                                      f"(its plan.json confirmatory): merge with the run's own freeze inputs")
            run = merge(args.out, the_plan)
            say(f"merged {run['items']} items, {run['rows']} rows into {args.out}")
            return 0
        if bool(args.endpoint) == bool(args.offline):
            raise BankReplayError("give exactly one of --endpoint (the store) and --offline (a store log)")
        if not (args.sources_dir or args.source_file) or not args.hierarchy:
            raise BankReplayError("--sources-dir (or --source-file for every source) and --hierarchy are required")
        if args.memory_gb > 0:
            cap = int(args.memory_gb * (1 << 30))
            resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        shard = parse_shard(args.shard)
        questions = read_question_ids(args.questions) if args.questions else None
        manifest_data, items = checked_items(args.manifest, frozen_bank_sha256(args.frozen_config), args.split,
                                             questions)
        graph = manifest_data.get("graph")
        if not graph:
            raise BankReplayError(f"{args.manifest} names no graph")
        say(f"{len(items)} checked items of split {args.split}"
            + (f" ({len(set(questions))} declared questions)" if questions else ""))
        names = [n for n in args.sources.split(",") if n]
        files = {k: Path(v) for k, _, v in (x.partition("=") for x in args.source_file)}
        unions = {}
        for spec in args.union:
            arm, _, srcs = spec.partition("=")
            unions[arm] = tuple(s for s in srcs.split(",") if s)
            if not unions[arm] or set(unions[arm]) - set(names):
                raise BankReplayError(f"--union {spec}: every source of a union must be one of --sources")
        t = time.time()
        hierarchy = SharedHierarchy.load(args.hierarchy)
        say(f"loaded the shared hierarchy in {time.time() - t:.1f}s")
        compilers, sources = {}, {}
        for name in names:
            path = source_path(args.sources_dir or Path("."), name, files)
            t = time.time()
            compilers[name] = CheckCompiler(load_source(path), hierarchy)
            sources[name] = {"path": str(path), "sha256": sha256_file(path), "digest": compilers[name].source.digest()}
            say(f"compiled {name} in {time.time() - t:.1f}s")
        if args.offline:
            store = {"offline": str(args.offline), "sha256": sha256_file(args.offline)}
            live, logs = None, read_store_logs(args.offline)
        else:
            # the store's identity from its own ?cmd=stats, in the plan: every shard and resume must find the same
            # index served, and each chunk checks it again before it is written
            identity = store_fingerprint(store_stats(args.endpoint))
            store = {"endpoint": args.endpoint, "timeout_s": args.timeout_s, "identity": identity}
            if args.store_json:
                described = json.loads(args.store_json.read_text(encoding="utf-8"))
                store["store_json"] = {"path": str(args.store_json), "sha256": sha256_file(args.store_json),
                                       "name": described.get("name"), "identity": described.get("identity")}
            live, logs = SparqlClient(args.endpoint, timeout_s=args.timeout_s, row_cap=CHECKER_ROW_CAP), None
        the_plan = plan(manifest=args.manifest, manifest_data=manifest_data, split=args.split, items=items,
                        questions=questions, chunk=args.chunk, sources=sources, unions=unions,
                        hierarchy={"path": str(args.hierarchy), "sha256": sha256_file(args.hierarchy),
                                   "hierarchy_sha256": hierarchy_sha256(hierarchy)},
                        store=store, code=code_commit(), confirmatory=confirmatory, shards=shard[1])
        open_run(args.out, the_plan)

        def check_store() -> None:
            now = store_fingerprint(store_stats(args.endpoint))
            if now != store["identity"]:
                raise BankReplayError(f"the store at {args.endpoint} now serves another index ({now}), not the plan's "
                                      f"({store['identity']}); nothing written for this chunk")

        def canary() -> None:  # a bounded data query: the store answers queries, not only ?cmd=stats
            live.query(CANARY)

        stats = store_stats(args.endpoint) if args.endpoint else None
        log_progress(args.out, shard, {"event": "start", "note": args.note, "store_stats": stats,
                                       "code": code_commit(), "preregistration": prereg})
        counts = run_chunks(args.out, the_plan, items, Arms(compilers, unions), graph, shard=shard, live_client=live,
                            store_logs=logs, check_store=check_store if args.endpoint else None,
                            canary=canary if args.endpoint else None,
                            authorised=confirmatory, say=say)
        log_progress(args.out, shard, {"event": "end", "counts": counts,
                                       "store_stats": store_stats(args.endpoint) if args.endpoint else None})
        say(f"shard {args.shard}: {counts}")
        if all_done(args.out, the_plan["n_chunks"]):
            run = merge(args.out, the_plan)
            say(f"merged {run['items']} items, {run['rows']} rows into {args.out}")
        else:
            say("other shards' chunks are not done yet: the last shard to finish merges (or run --merge-only)")
    except (BankReplayError, InputError, LedgerError, FileNotFoundError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
