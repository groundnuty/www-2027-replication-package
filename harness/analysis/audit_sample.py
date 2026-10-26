"""The post-run audit's sample (S3.1; pre-registration §5 "Audits", H4 and H6), drawn from a checker ledger by the
seeded rule of ``paper/audit-protocol.md`` (anonymous/science-repo, b9777bc: the drawing rules settled by
the study lead in the design log Oct 07:36, and change-log entries 127 and 128; seed 20261028).

Usage (the sample is drawn after the confirmatory runs, from the test ledger; a dev ledger only as a dry run, which
tests the code and builds no annotation package):

    uv run python -m harness.analysis.audit_sample --ledger LEDGER.jsonl --split test \\
        --config data/analysis/prereg-config.json --records OUT [OUT ...] --out SAMPLE.json \\
        [--private-out PRIVATE.json]
    uv run python -m harness.analysis.audit_sample --ledger DEV-LEDGER.jsonl --split dev --dry-run \\
        --records DEV-OUT --out /tmp/x.json

**Outputs.** An output is one run id of the ledger (harness/checkloop/ledger.py: a row per initial output, source and
use), its flags read from its ``replay`` rows: the offline checker is the registered definition (change-log entry
125), so an online row (use ``checker`` or ``both``) never decides a kind, and an output with no replay row is refused.
Rows of one run id must agree on its graph, model, cohort, question and correctness. Fault-bank rows (model or cohort
``bank``) and models outside the registered panel (the config's ``panel``) are not outputs; both are counted.

**Kinds** (``in_kind``):

* A, residue: wrong, and no primary source flagged it (``RESIDUE_SOURCES``, change-log entry 128: curated, census-t5
  and the generated draws, the natural and, on UniProt, the masked): no F on their replay rows. The census sensitivity
  thresholds (t2, t20), the union arm (C4's union, in code curated plus census), variants (``<arm>/<variant>``) and
  controls (``generic_retry/<arm>``) don't count. U (unknown) is not a flag.
* B, false flag: correct, and the curated source's replay row is F.
* C, shared premise: the curated-prompt cohort, wrong, and the curated source's replay row is P.

**Cells** (``CELLS``): A, 8 per graph x model group (UniProt, Rhea, opaque Rhea, corporate x Claude, open-weight), 64;
B, 8 per graph, 32; C, 16 from UniProt and Rhea pooled as one cell; so 112 public items. A private package's A and B
add 10 each, 20, whose graph is ``PRIVATE_GRAPHS``. The model group is the panel family: "claude", or open-weight for
every other family.

**The draw** (``draw``): the cells in ``CELLS``'s fixed order, A, then B, then C (within A, graph by graph in
``PUBLIC_GRAPHS``'s order, Claude before open-weight). In a cell (``draw_cell``), its base questions are ranked by
sha256("<seed>|<cell>|<base question id>") (ties by id); a base question already drawn in the same kind is skipped,
so no question appears twice in a kind, and the next ranked takes its place; the first <quota> that remain each give
their output of the least sha256("<seed>|<cell>|<base question id>|<run id>"). C excludes the outputs already drawn
for A (not their base questions). The sample does not depend on the ledger's row order, and it can be recomputed from
this text alone. A cell with fewer base questions than its quota is taken whole (one output per base question) and its
shortfall is reported, with what was skipped and excluded.

**Kind C's prompt and witness** (change-log entry 127). For C, annotators see the ontology text rendered into that
agent's prompt. A C item links to it from its own run record (``--records``: the run records the ledger was exported
from, never the graph's current prompt): ``prompt_condition`` and ``prompt_id``, ``prompt_sha256`` (the rendered
prompt), ``ontology_block_sha256`` (the block the prompt added to none-v1, ``data/prompts/blocks/<sha256>.txt`` in the
repository's content-addressed store, harness/agent/prompts.py) and ``record_sha256`` (the record as stored). The record
must agree with the ledger (graph, model, question, split), its prompt condition must be the item's cohort, its prompt
and agent config must name the same block, and the block must be in the repository's store with bytes that match its
name; otherwise nothing is written. A package built from the sample embeds the block's text by that name. A C item
also carries ``response`` (``C_RESPONSE``), the fields each annotator's copy fills independently:
``follows_from_ontology``, ``witness`` (the quoted statement the wrong answer follows from) and
``statement_wrong_or_misleading``; a ``yes`` without a quoted witness is coded ``unsure``. A and B items carry neither.

**The private package.** Rows of a private graph are drawn only into ``--private-out`` (mode 600, refused inside this
repository), and nothing of them, ids or counts, goes to the public file, stdout or stderr: a ledger holding them
without ``--private-out`` writes nothing. No ``--records`` path is written or printed either, since the OUTs given
include a private graph's; each C item's ``record_sha256`` pins the record it was linked from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.agent import prompts
from harness.checkloop.ledger import BANK, PLACEHOLDER, LedgerError, read_ledger
from harness.checkloop.matrix import BANK_MODEL, load_raw_records

ROOT = Path(__file__).resolve().parents[2]
AUDIT_SCHEMA = "h4-audit-sample/2"  # /2: kind C's prompt link and response (entry 127)
PROTOCOL = "anonymous/science-repo paper/audit-protocol.md at b9777bc"
SEED = 20261028  # audit-protocol.md "Sample (seed 20261028)"
PUBLIC_GRAPHS = ("uniprot-sp-2026_03", "rhea-142", "opaque-rhea-142", "text2sparql-ck26")
PRIVATE_GRAPHS = ("a15",)
CLAUDE, OPEN = "claude", "open-weight"
GROUPS = (CLAUDE, OPEN)
CLAUDE_FAMILY = "claude"  # the config's panel_families name of the Claude models
CURATED = "curated"  # the config's curated_source
CURATED_COHORT = "curated"  # the config's curated_prompt_cohort
REPLAY = "replay"
KINDS = ("A", "B", "C")  # also the order the kinds' cells are drawn in
# the primary sources (change-log entry 128; science, design log): curated, the census at threshold 5 and the generated
# draws, the natural ones and the masked ones (the config's masked_sources, UniProt's only); census t2 and t20 are
# sensitivity variants, and C4's union (curated plus census) is an arm, not a source
RESIDUE_SOURCES = ("curated", "census-t5", "generated-1", "generated-2", "generated-3",
                   "generated-dc-1", "generated-dc-2", "generated-dc-3")
OUTPUT_FIELDS = ("graph", "model", "cohort", "question_id", "base_question_id", "initial_correct")
# kind C's response, copied into each annotator's package (audit-protocol.md, codebook C; entry 127)
C_RESPONSE = {"follows_from_ontology": None, "witness": None, "statement_wrong_or_misleading": None}
PROMPT_LINK = ("prompt_condition", "prompt_id", "prompt_sha256", "ontology_block_sha256", "record_sha256")
RULES = {
    "outputs": "one per run id, its flags from its replay rows (entry 125); bank rows and models outside the panel are "
               "not outputs",
    "A": f"wrong, and no replay row of a primary source ({', '.join(RESIDUE_SOURCES)}) is F (entry 128); census t2 "
         "and t20, the union arm, variants and controls don't count; U is not a flag",
    "B": "correct, and the curated source's replay row is F",
    "C": "the curated-prompt cohort, wrong, and the curated source's replay row is P; outputs drawn for A excluded",
    "C_prompt": "each C item names the prompt its own run record shows (condition, id, the rendered prompt's sha256) "
                "and the ontology block it added, by sha256 in data/prompts/blocks, checked present with matching "
                "bytes; the record's sha256 beside them (entry 127)",
    "C_response": "per annotator: follows_from_ontology (yes, no or unsure), witness (the quoted statement), "
                  "statement_wrong_or_misleading (yes, no or unsure); a yes without a quoted witness is coded unsure",
    "group": "claude: the panel family 'claude'; open-weight: every other panel family",
    "draw": "cells in a fixed order, A then B then C; per cell, base questions by sha256('<seed>|<cell>|<base question "
            "id>') (ties by id), skipping one already drawn in the same kind, the first <quota>; for each, its output "
            "of the least sha256('<seed>|<cell>|<base question id>|<run id>'); a cell with fewer base questions is "
            "taken whole and its shortfall reported",
}


class AuditError(ValueError):
    """A ledger or a request the sampler refuses. Messages name run ids and fields, never a row's content."""


@dataclass(frozen=True)
class Cell:
    kind: str
    name: str
    graphs: tuple[str, ...]
    group: str | None  # None: both model groups
    quota: int
    private: bool = False


# the protocol's fixed order, A, then B, then C (draw); the private cells come last in their kind, and are drawn apart
CELLS: tuple[Cell, ...] = (
    *(Cell("A", f"A/{g}/{grp}", (g,), grp, 8) for g in PUBLIC_GRAPHS for grp in GROUPS),
    *(Cell("A", f"A/{g}", (g,), None, 10, private=True) for g in PRIVATE_GRAPHS),
    *(Cell("B", f"B/{g}", (g,), None, 8) for g in PUBLIC_GRAPHS),
    *(Cell("B", f"B/{g}", (g,), None, 10, private=True) for g in PRIVATE_GRAPHS),
    Cell("C", "C/uniprot-sp-2026_03+rhea-142", ("uniprot-sp-2026_03", "rhea-142"), None, 16),
)


@dataclass(frozen=True)
class Output:
    run_id: str
    graph: str
    model: str
    group: str
    cohort: str
    question_id: str
    base_question_id: str
    initial_correct: bool
    letters: tuple[tuple[str, str], ...]  # (source, status letter) of its replay rows, sorted by source

    def letter(self, source: str) -> str | None:
        return dict(self.letters).get(source)


@dataclass(frozen=True)
class CellDraw:
    cell: Cell
    population_outputs: int  # the cell's outputs of its kind, before anything is excluded or skipped
    population_base_questions: int
    items: tuple[Output, ...]  # in the draw's order
    excluded_outputs: int = 0  # C: outputs already drawn for A
    skipped_base_questions: int = 0  # already drawn in the same kind (counted over the remaining outputs)

    @property
    def shortfall(self) -> int:
        return self.cell.quota - len(self.items)


def counts_for_residue(source: str) -> bool:
    """A primary source (``RESIDUE_SOURCES``), whose flag takes an output out of A."""
    return source in RESIDUE_SOURCES


def in_kind(kind: str, o: Output) -> bool:
    if kind == "A":
        return not o.initial_correct and not any(s == "F" for src, s in o.letters if counts_for_residue(src))
    if kind == "B":
        return o.initial_correct and o.letter(CURATED) == "F"
    if kind == "C":
        return o.cohort == CURATED_COHORT and not o.initial_correct and o.letter(CURATED) == "P"
    raise AuditError(f"no audit kind {kind!r}")


def group_of(model: str, families: Mapping[str, str]) -> str | None:
    """The model group of a panel model; None for a model outside the panel."""
    if model not in families:
        return None
    return CLAUDE if families[model] == CLAUDE_FAMILY else OPEN


def families_of(config: Mapping[str, Any]) -> dict[str, str]:
    """The panel's model -> family from a pre-registration config (data/analysis/prereg-config.json)."""
    panel, fams = list(config.get("panel") or ()), list(config.get("panel_families") or ())
    if not panel or len(panel) != len(fams) or PLACEHOLDER in panel + fams:
        raise AuditError("the config must name every panel model and one family per model, none of them [FILL]")
    return dict(zip(panel, fams, strict=True))


def collect_outputs(rows: Iterable[Mapping[str, Any]], families: Mapping[str, str]) -> tuple[list[Output], Counter]:
    """The ledger's outputs (module docstring), sorted by run id, and what was not an output: ``bank_rows``,
    ``online_rows`` (rows of outputs that decide nothing) and ``outputs_outside_panel``."""
    excluded: Counter = Counter()
    by_run: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        if r["model"] == BANK_MODEL or r["cohort"] == BANK:
            excluded["bank_rows"] += 1
            continue
        by_run[r["run_id"]].append(r)
    outputs = []
    for run_id in sorted(by_run):
        rs = by_run[run_id]
        for f in OUTPUT_FIELDS:
            if len({json.dumps(r[f]) for r in rs}) > 1:
                raise AuditError(f"the rows of output {run_id} disagree on {f}")
        replay = [r for r in rs if r["use"] == REPLAY]
        excluded["online_rows"] += len(rs) - len(replay)
        if not replay:
            raise AuditError(f"output {run_id} has no replay row: its flags are the offline checker's (entry 125)")
        sources = Counter(r["source"] for r in replay)
        if dup := sorted(s for s, n in sources.items() if n > 1):
            raise AuditError(f"output {run_id} has two replay rows of source {dup[0]}")
        first = rs[0]
        group = group_of(first["model"], families)
        if group is None:
            excluded["outputs_outside_panel"] += 1
            continue
        outputs.append(Output(run_id, first["graph"], first["model"], group, first["cohort"], first["question_id"],
                              first["base_question_id"], bool(first["initial_correct"]),
                              tuple(sorted((r["source"], r["status"]) for r in replay))))
    return outputs, excluded


def _rank(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


def draw_cell(cell: Cell, outputs: Iterable[Output], seed: int = SEED, *, skip_bases: Iterable[str] = (),
              exclude_runs: Iterable[str] = ()) -> CellDraw:
    """The cell's draw by the seeded rule (module docstring): ``skip_bases`` are the base questions already drawn in
    its kind, ``exclude_runs`` the outputs already drawn for an earlier kind that this one excludes."""
    skip, exclude = set(skip_bases), set(exclude_runs)
    members = [o for o in outputs
               if o.graph in cell.graphs and (cell.group is None or o.group == cell.group) and in_kind(cell.kind, o)]
    by_base: dict[str, list[Output]] = defaultdict(list)
    for o in members:
        if o.run_id not in exclude:
            by_base[o.base_question_id].append(o)
    skipped = skip & set(by_base)
    ranked = sorted(set(by_base) - skipped, key=lambda b: (_rank(seed, cell.name, b), b))
    items = tuple(min(by_base[b], key=lambda o: (_rank(seed, cell.name, b, o.run_id), o.run_id))
                  for b in ranked[:cell.quota])
    return CellDraw(cell, len(members), len({o.base_question_id for o in members}), items,
                    excluded_outputs=sum(o.run_id in exclude for o in members), skipped_base_questions=len(skipped))


def draw(outputs: Sequence[Output], cells: Sequence[Cell] = CELLS, seed: int = SEED) -> list[CellDraw]:
    """The cells in the order given, which must be A's, then B's, then C's (``CELLS``'s order is the protocol's): a
    base question drawn in a kind is skipped in its later cells, and C excludes the outputs drawn for A."""
    order = [KINDS.index(c.kind) for c in cells]
    if order != sorted(order):
        raise AuditError(f"the cells are drawn A, then B, then C, not {[c.name for c in cells]}")
    drawn_bases: dict[str, set[str]] = defaultdict(set)
    drawn_for_a: set[str] = set()
    out = []
    for c in cells:
        d = draw_cell(c, outputs, seed, skip_bases=drawn_bases[c.kind],
                      exclude_runs=drawn_for_a if c.kind == "C" else ())
        drawn_bases[c.kind].update(o.base_question_id for o in d.items)
        if c.kind == "A":
            drawn_for_a.update(o.run_id for o in d.items)
        out.append(d)
    return out


def summary(draws: Sequence[CellDraw], outputs: Sequence[Output], excluded: Mapping[str, int]) -> dict[str, Any]:
    """Counts beside the draw: what was excluded, the outputs per graph (and those of no audited graph, or without a
    curated replay row), the primary sources each graph's outputs have rows of, the totals, the base
    questions skipped per kind, and two checks of the rules that must read 0: a base question drawn twice in a kind,
    an output drawn in two kinds."""
    audited = {g for d in draws for g in d.cell.graphs}
    cells_of: dict[str, dict[str, set[str]]] = {k: defaultdict(set) for k in KINDS}
    kinds_of: dict[str, set[str]] = defaultdict(set)
    for d in draws:
        for o in d.items:
            cells_of[d.cell.kind][o.base_question_id].add(d.cell.name)
            kinds_of[o.run_id].add(d.cell.kind)
    pairs = [(a, b) for i, a in enumerate(KINDS) for b in KINDS[i + 1:]]
    return {
        "excluded": dict(sorted(excluded.items())),
        "outputs_by_graph": dict(sorted(Counter(o.graph for o in outputs if o.graph in audited).items())),
        "outputs_of_no_audited_graph": dict(sorted(Counter(o.graph for o in outputs if o.graph not in audited).items())),
        "outputs_without_curated_row": dict(sorted(Counter(o.graph for o in outputs
                                                           if o.letter(CURATED) is None).items())),
        "residue_sources_by_graph": {g: sorted({s for o in outputs if o.graph == g for s, _ in o.letters
                                                if counts_for_residue(s)})
                                     for g in sorted({o.graph for o in outputs})},
        "quota": sum(d.cell.quota for d in draws),
        "drawn": sum(len(d.items) for d in draws),
        "shortfall": sum(d.shortfall for d in draws),
        "skipped_base_questions": {k: sum(d.skipped_base_questions for d in draws if d.cell.kind == k) for k in KINDS},
        "excluded_outputs_drawn_for_a": sum(d.excluded_outputs for d in draws),
        "repeated_base_questions": {k: sum(len(c) > 1 for c in cells_of[k].values()) for k in KINDS},
        "outputs_in_two_kinds": {f"{a}+{b}": sum({a, b} <= ks for ks in kinds_of.values()) for a, b in pairs},
    }


def prompts_of(items: Sequence[Output], raw_records: Iterable[str], split: str) -> dict[str, dict[str, Any]]:
    """Kind C's link to the ontology text each item's agent was shown (module docstring), by run id, from the item's
    own run record among ``raw_records`` (records as stored, e.g. ``matrix.load_raw_records``; read one at a time and
    kept only for these items). Every refusal names the run id and the field, never a record's content."""
    wanted = {o.run_id: o for o in items}
    found: dict[str, tuple[dict[str, Any], str]] = {}
    for raw in raw_records:
        rec = json.loads(raw)
        rid = rec.get("run_id")
        if rid not in wanted:
            continue
        if rid in found:
            raise AuditError(f"output {rid} has two run records")
        found[rid] = (rec, hashlib.sha256(raw.encode("utf-8")).hexdigest())
    linked = {}
    for rid in sorted(wanted):
        o = wanted[rid]
        if rid not in found:
            raise AuditError(f"output {rid} (kind C) has no run record among --records")
        rec, record_sha256 = found[rid]
        seen = {"graph": (rec.get("store") or {}).get("name"), "model": rec.get("model"),
                "question_id": rec.get("question_id"), "base_question_id": rec.get("base_question_id"),
                "split": rec.get("split")}
        want = {"graph": o.graph, "model": o.model, "question_id": o.question_id,
                "base_question_id": o.base_question_id, "split": split}
        if bad := sorted(k for k in want if seen[k] != want[k]):
            raise AuditError(f"output {rid}: its run record disagrees with the ledger on {', '.join(bad)}")
        prompt = rec.get("prompt") or {}
        if prompt.get("condition") != o.cohort:
            raise AuditError(f"output {rid}: its run record's prompt.condition is not its cohort {o.cohort!r}")
        block = prompt.get("ontology_block_sha256")
        if block is None:
            raise AuditError(f"output {rid}: its run record names no ontology block (prompt.ontology_block_sha256)")
        if (rec.get("agent_config") or {}).get("ontology_block_sha256") != block:
            raise AuditError(f"output {rid}: its run record's prompt and agent_config name different ontology blocks")
        try:  # the repository's store only: a further store (a private graph's) is never a public item's link
            prompts.load_block(block, prompts.BLOCK_DIR)
        except (OSError, ValueError) as e:
            raise AuditError(f"output {rid}: its ontology block is not in the repository's block store with bytes "
                             f"that match its sha256 ({type(e).__name__})") from None
        linked[rid] = {"prompt_condition": prompt["condition"], "prompt_id": prompt.get("id"),
                       "prompt_sha256": prompt.get("sha256"), "ontology_block_sha256": block,
                       "record_sha256": record_sha256}
    return linked


def _cell_entry(d: CellDraw) -> dict[str, Any]:
    return {"cell": d.cell.name, "kind": d.cell.kind, "graphs": list(d.cell.graphs), "group": d.cell.group,
            "quota": d.cell.quota, "population_outputs": d.population_outputs,
            "population_base_questions": d.population_base_questions, "excluded_outputs": d.excluded_outputs,
            "skipped_base_questions": d.skipped_base_questions, "drawn": len(d.items), "shortfall": d.shortfall}


def _item_entry(d: CellDraw, o: Output, linked: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    entry = {"kind": d.cell.kind, "cell": d.cell.name, "graph": o.graph, "model_group": o.group, "model": o.model,
             "cohort": o.cohort, "run_id": o.run_id, "question_id": o.question_id,
             "base_question_id": o.base_question_id}
    if d.cell.kind == "C":
        if o.run_id not in linked:
            raise AuditError(f"output {o.run_id} (kind C) has no link to the ontology block its prompt added")
        entry.update({k: linked[o.run_id][k] for k in PROMPT_LINK})
        entry["response"] = dict(C_RESPONSE)
    return entry


def sample_document(header: Mapping[str, Any], draws: Sequence[CellDraw], outputs: Sequence[Output],
                    excluded: Mapping[str, int], linked: Mapping[str, Mapping[str, Any]] | None = None
                    ) -> dict[str, Any]:
    """The sample: ``linked`` is kind C's prompt link by run id (``prompts_of``), which every C item needs."""
    linked = linked or {}
    return {"header": dict(header), "cells": [_cell_entry(d) for d in draws],
            "items": [_item_entry(d, o, linked) for d in draws for o in d.items],
            "summary": summary(draws, outputs, excluded)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dump(doc: Mapping[str, Any]) -> str:
    return json.dumps(doc, indent=1, sort_keys=True) + "\n"


def _write_private(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)


def check_request(split: str | None, wanted: str, dry_run: bool, exploratory: bool) -> None:
    """A registered draw is of the test ledger; a dev ledger (or an exploratory export) only as a dry run."""
    if split is None:
        raise AuditError("the ledger names no split (checker-ledger/1); the audit draws from a checker-ledger/2 file")
    if split != wanted:
        raise AuditError(f"the ledger's split is {split!r}, not --split {wanted!r}")
    if split == "test" and dry_run:
        raise AuditError("a test ledger is drawn only as the registered sample, never as a dry run")
    if split != "test" and not dry_run:
        raise AuditError(f"a {split} ledger is drawn only as a dry run (--dry-run), to test the code")
    if exploratory and not dry_run:
        raise AuditError("an exploratory ledger is drawn only as a dry run")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Draw the post-run audit sample from a checker ledger (seed 20261028).")
    ap.add_argument("--ledger", type=Path, required=True, help="a checker-ledger/2 file (plain JSON lines)")
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--config", type=Path, default=ROOT / "data" / "analysis" / "prereg-config.json",
                    help="the pre-registration config naming the panel and its families")
    ap.add_argument("--records", type=Path, nargs="+", default=[],
                    help="the run records the ledger was exported from (OUT directories or records archives), read "
                         "for the prompt of each kind C item; needed when C draws an item")
    ap.add_argument("--out", type=Path, required=True, help="the public sample (JSON)")
    ap.add_argument("--private-out", type=Path, help="the private package's sample (mode 600, outside the repository)")
    ap.add_argument("--dry-run", action="store_true", help="a dev ledger's draw, to test the code")
    args = ap.parse_args(argv)
    try:
        ledger = read_ledger(args.ledger)
        check_request(ledger.split, args.split, args.dry_run, ledger.exploratory)
        families = families_of(json.loads(args.config.read_text(encoding="utf-8")))
        private_rows = [r for r in ledger.rows if r["graph"] in PRIVATE_GRAPHS]
        public_rows = [r for r in ledger.rows if r["graph"] not in PRIVATE_GRAPHS]
        private_out = args.private_out.resolve() if args.private_out else None
        if private_rows and private_out is None:
            raise AuditError("the ledger holds rows of a private graph: give --private-out; nothing is written")
        if private_out is not None and private_out.is_relative_to(ROOT):
            raise AuditError("--private-out must be outside the repository")
        # no --records path is printed or written: the registered draw is given a private graph's OUT too, and each C
        # item's record_sha256 pins the record it read
        if missing := [i for i, p in enumerate(args.records, 1) if not p.exists()]:
            raise AuditError(f"--records path {missing[0]} of {len(args.records)} does not exist")
        header = {"schema": AUDIT_SCHEMA, "protocol": PROTOCOL, "seed": SEED, "split": ledger.split,
                  "dry_run": args.dry_run, "exploratory": ledger.exploratory, "rules": RULES,
                  "ledger": {"path": str(args.ledger.resolve()), "schema": ledger.schema,
                             "sha256": _sha256(args.ledger)},
                  "config": {"path": str(args.config.resolve()), "sha256": _sha256(args.config)}}
        public_cells = [c for c in CELLS if not c.private]
        outputs, excluded = collect_outputs(public_rows, families)
        draws = draw(outputs, public_cells)
        c_items = [o for d in draws if d.cell.kind == "C" for o in d.items]
        if c_items and not args.records:
            raise AuditError(f"kind C drew {len(c_items)} items: give --records, the run records that name the "
                             "ontology block each item's agent was shown")
        linked = prompts_of(c_items, load_raw_records(args.records), ledger.split) if c_items else {}
        public_doc = sample_document({**header, "private": False}, draws, outputs, excluded, linked)
        private_doc = None
        if private_out is not None:
            try:
                p_outputs, p_excluded = collect_outputs(private_rows, families)
            except AuditError:  # its message could name a private run id
                raise AuditError("the private graph's rows are refused; run the check inside the private tree") from None
            p_draws = draw(p_outputs, [c for c in CELLS if c.private])
            private_doc = sample_document({**header, "private": True}, p_draws, p_outputs, p_excluded)
    except (AuditError, LedgerError, OSError, ValueError) as e:
        print(f"audit sample refused: {e}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(_dump(public_doc), encoding="utf-8")
    if private_doc is not None:
        _write_private(private_out, _dump(private_doc))
    for c in public_doc["cells"]:
        print(f"{c['cell']}: quota {c['quota']}, population {c['population_outputs']} outputs on "
              f"{c['population_base_questions']} base questions, excluded {c['excluded_outputs']} (drawn for A), "
              f"skipped {c['skipped_base_questions']} base questions, drawn {c['drawn']}, shortfall {c['shortfall']}")
    s = public_doc["summary"]
    print(f"public: drawn {s['drawn']} of {s['quota']} (shortfall {s['shortfall']}); skipped base questions "
          f"{s['skipped_base_questions']}; repeated base questions {s['repeated_base_questions']}; outputs in two "
          f"kinds {s['outputs_in_two_kinds']}; excluded {s['excluded']}")
    if private_doc is not None:
        print("the private cells are in --private-out only (no ids or counts printed)")
    print(f"wrote {args.out}{' (dry run)' if args.dry_run else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
