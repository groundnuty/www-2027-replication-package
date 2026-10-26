"""The generated source of a graph whose census and instance sample were built elsewhere (by
scripts/build_census_onegraph.py, which reads the store), by the recipe of every other graph (harness.sources.generated):
the same bundle, request template, fit rule, batch runner, retry rule and draft parser. Only the inputs differ: they are
read from files, never from a store.

    python -I scripts/generated_from_files.py bundle --census <census-t5.json> --census-sha256 <hex> \\
        --census-content-digest <hex> --sample <sample.json> --sample-sha256 <hex> --sources <private dir>
    python -I scripts/generated_from_files.py estimate --sources <private dir> --outputs 5000 13000 45000
    python -I scripts/generated_from_files.py draw --sources <private dir> [--retry]
    python -I scripts/generated_from_files.py drafts --sources <private dir>

``bundle`` refuses any input whose sha256 (and the census's content digest) is not the one given, copies the census
byte for byte to <sources>/census-t5.json (where the draw command checks the bundle against it, ``census_current``),
builds the bundle with rule 33's fit counted on the exact request (the free counting endpoint) and writes it under
<sources>/generated/input/, with staging.json recording what was staged. ``estimate`` prices the three draws from the
recorded count. ``draw`` runs ``generated.run_batch_draws``. ``drafts`` writes generated-<i>.json for every successful
draw, each made from the current bundle, parsed with the census's graph and membership. Every directory is mode 700 and
every file mode 600; stdout carries counts, hashes and costs only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_census_onegraph import content_digest as builder_content_digest

from harness.sources import generated as gen
from harness.sources.model import OntologySource
from harness.sources.tokens import count_tokens

CENSUS_FILE = f"census-t{gen.DRAW_THRESHOLD}.json"
KEY_FILE = Path.home() / ".config" / "www-2027" / "anthropic_api_key"


class InputRefused(RuntimeError):
    pass


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _private_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)
    return path


def _privatise(root: Path) -> None:
    """Mode 700 for every directory and 600 for every file under ``root``."""
    _private_dir(root)
    for p in root.rglob("*"):
        p.chmod(0o700 if p.is_dir() else 0o600)


@contextmanager
def private_writes(root: Path):
    """Every write under ``root`` is private: the umask is 077 while it runs (the recipe's own writers create files),
    what is already there is made private before anything is written over it, and everything again afterwards, on a
    failure too."""
    old = os.umask(0o077)
    try:
        _privatise(root)
        yield root
    finally:
        try:
            _privatise(root)
        finally:
            os.umask(old)


def _write_private(path: Path, data: bytes) -> None:
    """Write ``data`` to a file that is mode 600 before the first byte lands, whatever mode it had."""
    _private_dir(path.parent)
    if path.exists():
        path.chmod(0o600)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)


def read_inputs(census: Path, census_sha256: str, census_content_digest: str, sample: Path,
                sample_sha256: str) -> tuple[bytes, OntologySource, gen.InstanceSample]:
    """The census file's bytes, the census and the sample, each read once and refused unless its bytes have the given
    sha256, and the census also unless it has the given content digest as its builder records it
    (``build_census_onegraph.content_digest``: the record without its provenance)."""
    raw = census.read_bytes()
    if _sha(raw) != census_sha256:
        raise InputRefused(f"census file sha256 {_sha(raw)} is not the recorded {census_sha256}")
    src = OntologySource.from_json(json.loads(raw))
    if builder_content_digest(src) != census_content_digest:
        raise InputRefused(f"census content digest {builder_content_digest(src)} is not the recorded "
                           f"{census_content_digest}")
    if src.name != CENSUS_FILE.removesuffix(".json") or src.support_threshold != gen.DRAW_THRESHOLD:
        raise InputRefused("the census file is not rule 33's drawable census (its name or threshold)")
    raw_sample = sample.read_bytes()
    if _sha(raw_sample) != sample_sha256:
        raise InputRefused(f"sample file sha256 {_sha(raw_sample)} is not the recorded {sample_sha256}")
    payload = json.loads(raw_sample)
    return raw, src, gen.InstanceSample(payload["spec"], payload["instances"])


def stage_and_build(census: Path, census_sha256: str, census_content_digest: str, sample: Path, sample_sha256: str,
                    sources: Path, count) -> dict:
    """Gate, stage and build: see the module docstring. ``count(bundle) -> input tokens`` counts the exact request.
    The staged census is the very bytes the gate checked."""
    raw, src, smp = read_inputs(census, census_sha256, census_content_digest, sample, sample_sha256)
    staged = sources / CENSUS_FILE
    if staged.exists() and staged.read_bytes() != raw:
        raise InputRefused(f"{staged} already holds another census file (sha256 {_sha(staged.read_bytes())}); a new "
                           "census needs a new sources directory")
    bundle, fit = gen.fit_bundle(src, smp, count)
    with private_writes(sources):
        if not staged.exists():
            _write_private(staged, raw)
        gen_dir = _private_dir(sources / "generated")
        gen.write_bundle(bundle, gen_dir, count_tokens(bundle["system"]) + count_tokens(bundle["user"]))
        _write_private(sources / "staging.json", (json.dumps({
            "census": {"from": str(census), "sha256": census_sha256, "content_digest": census_content_digest,
                       "digest": src.digest(), "staged_as": CENSUS_FILE},
            "sample": {"from": str(sample), "sha256": sample_sha256, "n_strata": smp.spec.get("n_strata"),
                       "n_instances": smp.spec.get("n_instances")},
            "request_sha256": bundle["request_sha256"], "user_sha256": bundle["user_sha256"], "input_fit": fit,
        }, indent=2, sort_keys=True) + "\n").encode())
    return bundle


def estimate(input_tokens: int, outputs: list[int]) -> dict:
    """USD for the three draws at ``generated.PRICES``: draw 1 writes the whole input to the 1-hour cache, draws 2 and
    3 read it, each writing ``output`` tokens; ``upper_bound``: every draw misses the cache (and so writes it again)
    and stops at max_tokens."""
    def cost(write: int, read: int, out: int) -> float:
        return gen.draw_cost({"cache_creation": {"ephemeral_1h_input_tokens": write, "ephemeral_5m_input_tokens": 0},
                              "cache_read_input_tokens": read, "output_tokens": out})

    by_output = {}
    for out in outputs:
        d1, d2 = cost(input_tokens, 0, out), cost(0, input_tokens, out)
        by_output[str(out)] = {"draw_1": d1, "draw_2": d2, "draw_3": d2, "total": round(d1 + 2 * d2, 4)}
    top = gen.PARAMETERS["max_tokens"]
    return {"input_tokens": input_tokens, "prices_usd_per_mtok": gen.PRICES, "by_output": by_output,
            "upper_bound": round(gen.N_DRAWS * cost(input_tokens, 0, top), 4), "upper_bound_output_tokens": top}


def draw(sources: Path, client, poll_s: float = 60.0, log=print, retry: bool = False) -> list[dict]:
    gen_dir = sources / "generated"
    with private_writes(sources):
        return gen.run_batch_draws(gen.load_bundle(gen_dir), gen_dir, client, poll_s, log=log, retry=retry,
                                   sources_dir=sources)


def draw_line(record: dict) -> str:
    """One draw's stdout line. A failed draw's status names the reason, which can quote the private reply (a parser
    error), so only "ok" or "failed" is printed; the reason stays in the private record."""
    status = "ok" if record.get("status") == "ok" else "failed"
    return f"draw {record['draw']}: {status} usage={record.get('usage')} cost=${record.get('cost_usd')}"


def write_drafts(sources: Path) -> list[Path]:
    """generated-<i>.json for every successful draw, written only when the bundle is current against the staged
    census (``generated.census_current``) and every draw was made from that bundle."""
    gen_dir = sources / "generated"
    bundle = gen.load_bundle(gen_dir)
    current, why = gen.census_current(bundle, sources, gen_dir)
    if not current:
        raise RuntimeError(f"no draft written: {why}")
    census = OntologySource.from_json(json.loads((sources / CENSUS_FILE).read_bytes()))
    drafts = []
    for i in range(1, gen.N_DRAWS + 1):
        src = gen.load_draw(gen_dir, i, census.graph, census.membership)
        if src is None:
            continue
        drawn_from = src.provenance.get("draw", {}).get("request_sha256")
        if drawn_from != bundle["request_sha256"]:
            raise RuntimeError(f"draw {i} was made from bundle {drawn_from}, not the current {bundle['request_sha256']}")
        drafts.append(src)
    out = []
    with private_writes(sources):
        for src in drafts:
            path = sources / f"{src.name}.json"
            _write_private(path, (json.dumps(src.to_json(), sort_keys=True, ensure_ascii=False,
                                             separators=(",", ":")) + "\n").encode())
            out.append(path)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bundle")
    for name in ("census", "sample"):
        b.add_argument(f"--{name}", type=Path, required=True)
        b.add_argument(f"--{name}-sha256", required=True)
    b.add_argument("--census-content-digest", required=True)
    e = sub.add_parser("estimate")
    e.add_argument("--outputs", type=int, nargs="+", required=True, help="output tokens per draw to price")
    d = sub.add_parser("draw")
    d.add_argument("--poll", type=float, default=60.0)
    d.add_argument("--retry", action="store_true", help="redraw failed draws with the identical request")
    sub.add_parser("drafts")
    for p in (b, e, d):
        p.add_argument("--key-file", type=Path, default=KEY_FILE)
    for p in sub.choices.values():
        p.add_argument("--sources", type=Path, required=True)
    args = ap.parse_args(argv)
    umask = os.umask(0o077)
    try:
        return run(args)
    except InputRefused as exc:  # its message is hashes and paths only
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - a traceback can quote private input or replies: it goes to a private file
        log, where = args.sources / "error.log", ""
        try:
            _write_private(log, traceback.format_exc().encode())
            where = f"; traceback in {log}"
        except OSError as werr:  # never re-raised, so no chained traceback reaches stderr
            where = f"; the traceback could not be written ({type(werr).__name__})"
        print(f"failed: {type(exc).__name__}{where}", file=sys.stderr)
        return 1
    finally:
        os.umask(umask)


def run(args) -> int:
    if args.cmd == "bundle":
        api = gen.anthropic_client(args.key_file)
        bundle = stage_and_build(args.census, args.census_sha256, args.census_content_digest, args.sample,
                                 args.sample_sha256, args.sources, lambda bd: gen.count_input(bd, api)["input_tokens"])
        print(json.dumps({"request_sha256": bundle["request_sha256"], "user_sha256": bundle["user_sha256"],
                          "status": bundle["status"], "input_fit": bundle["input_fit"],
                          "n_strata": bundle["sample_spec"].get("n_strata"),
                          "n_instances": bundle["sample_spec"].get("n_instances")}, sort_keys=True))
    elif args.cmd == "estimate":
        fit = gen.load_bundle(args.sources / "generated").get("input_fit") or {}
        if not fit.get("tried"):
            raise SystemExit("the bundle records no count")
        print(json.dumps(estimate(fit["tried"][-1]["input_tokens"], args.outputs), indent=2, sort_keys=True))
    elif args.cmd == "draw":
        records = draw(args.sources, gen.anthropic_client(args.key_file), args.poll,
                       log=lambda m: print(m, flush=True), retry=args.retry)
        for r in records:
            print(draw_line(r), flush=True)
    else:
        for path in write_drafts(args.sources):
            src = OntologySource.from_json(json.loads(path.read_bytes()))
            print(f"{path.name}: sha256 {_sha(path.read_bytes())} digest {src.digest()} classes {len(src.classes)} "
                  f"properties {len(src.properties)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
