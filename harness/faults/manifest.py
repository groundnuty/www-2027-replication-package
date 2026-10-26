"""The fault bank manifest for one graph (issue E1.3, criteria 1, 5 and 7).

Base: the gold query of every question, with its dev or test split (test is the primary report).
Every gold is kept untouched as a control (false flags). Every answer-changing mutation is kept,
with one denominator common to all sources; answer-neutral mutations are kept as controls;
mutations that are answer-neutral but violate some source's axioms are marked conformance-only and
analysed separately. Rejections (rdflib or store parse errors, time-outs, server errors, invented
identifiers found present) are kept in the item list and counted per operator.

Descriptive strata (design ruling (d) in the design log, change-log entry 63): the type-consistent
constant swap's items are kept with bank ``descriptive`` (their answer effect, always
answer-changing, stays in ``answer_effect``), so W, C2's type-wrong set, the controls and the
wrong-item cap, which read ``bank``, never take them; the summary's ``estimand_sets`` states the sets.

Every item carries ``internal_node`` (ruling (c)): whether the constant it introduces is a
store-internal node by the constants check's rule (``shapes.internal``). C2 (ii) and the type-wrong
rates are reported with and without those items; the primary keeps them.

The build writes no timestamp and orders everything, so the same inputs give the same bytes.
Freezing (design log: before any checker runs on the bank): a build with ``freeze`` marks its manifest
frozen, which only a final build of pinned inputs may do, and records every input's sha256, the store
and the data files' hashes; committing it is the freeze record. A directory that holds a frozen
manifest is never rewritten: a rebuild into it only checks that every data file and every input is
the same, and is refused (``FrozenBankChanged``, naming the changed items) otherwise.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from harness.faults.conformance import Labeller, label_all
from harness.faults.operators import (
    ALL_OPERATORS,
    DESCRIPTIVE,
    FAULT_CLASS,
    OPERATORS,
    SWAPS,
    TYPE_CONSISTENT,
    Context,
    Mutation,
    Skip,
    generate,
    type_consistent_swaps,
)
from harness.faults.query import RDF_TYPE, RDFS_SUBCLASSOF, ParsedQuery, parse
from harness.faults.rewrites import REWRITES, RewriteSettings, draw
from harness.faults.shapes import FORMS, form, internal
from harness.faults.store import Execution, Runner, answer_effect, bank_rules, query_sha
from harness.score.rules import ScoringRules
from harness.score.scorer import score_rows
from harness.terms import row_from_json

SCHEMA = "fault-bank/1"
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class GraphProfile:
    name: str
    data_graph: str
    stale_sources: tuple[str, ...]
    frozen_questions_sha256: str | None = None
    stale_note: str = ""
    membership: str = RDF_TYPE  # the graph's membership predicate (rule 20's convention), for the rewrites
    # a second question source: bank-only gold queries with no question text (``question: null``), pinned
    # by their own file's sha256 (design log); None: the graph has none, and none may be added
    extension_sha256: str | None = None


PROFILES = {
    "rhea-142": GraphProfile(
        "rhea-142", "https://sparql.rhea-db.org/rhea", ("obo_replaced_by", "rhea_obsolete_comment"),
        membership=RDFS_SUBCLASSOF,
        # the frozen round-2 set (feat/qgen b50e77d, generator 0.4.3); round 1 was 00e84760 (feat/qgen 5469ab5)
        frozen_questions_sha256="b204846e7c20e82971ec65590298d25d54dd480efcea4f178e5044847dfa96b7",
        stale_note=("ChEBI: 18,802 classes marked owl:deprecated with IAO_0100001 'term replaced by' (the stale "
                    "term exists in the store only as a deprecation record, in no source and not in the shared "
                    "hierarchy). Rhea: reactions with status rh:Obsolete name their replacement only in rdfs:comment "
                    "('replaced by RHEA:n'); they keep their type and accession (obsolete_entity). Rhea compounds, "
                    "schema classes and predicates: no older release or deprecation record in the store, so none.")),
    "uniprot-sp-2026_03": GraphProfile(
        "uniprot-sp-2026_03", "http://sparql.uniprot.org/uniprot",
        ("obo_replaced_by", "rhea_obsolete_comment", "uniprot_replaces"),
        # the frozen round-2 set (feat/qgen-uniprot 0b247dd; round 1 was b805acd9), and the bank-only
        # extension (feat/qgen-extension b3d0af7, extension/questions-ext.jsonl, generator 0.3.0-ext1)
        frozen_questions_sha256="17eef284d7c89c64e5da42245b0b09f21215d2ebd01e086670057baae36c9184",
        extension_sha256="f11dff07f69dffc934e148d3686b3ecfd9f6d3c9bd7088baffe9c3c39315cff8",
        stale_note=("Candidate sources, to be confirmed by light probes when the census lands: ChEBI and GO "
                    "deprecations (owl:deprecated with IAO_0100001), Rhea obsolete reactions (the rhea graph), "
                    "secondary accessions (up:replaces in the uniprot graph; obsolete entries are not loaded), "
                    "and core.owl's 60 owl:deprecated terms, a few with rdfs:seeAlso to their replacement "
                    "(a static map; core is the compared source, so a stale core term is absent from census and "
                    "generated sources by construction).")),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_questions(profile: GraphProfile, questions_sha: str, *, unfrozen: bool = False) -> bool:
    """Whether the question file is the graph's frozen set; a different file is refused unless the
    caller declares it pre-freeze (``unfrozen``, a provisional bank only)."""
    frozen = not profile.frozen_questions_sha256 or questions_sha == profile.frozen_questions_sha256
    if not frozen and not unfrozen:
        raise ValueError(f"{profile.name}: question file sha256 {questions_sha} is not the frozen "
                         f"{profile.frozen_questions_sha256}")
    return frozen


def check_extension(profile: GraphProfile, extension_sha: str, *, unfrozen: bool = False) -> bool:
    """Whether the extension file is the graph's pinned one; a different file is refused unless the
    caller declares it pre-freeze (``unfrozen``), and a graph that pins none takes none."""
    if profile.extension_sha256 is None:
        raise ValueError(f"{profile.name} pins no bank-only extension")
    pinned = extension_sha == profile.extension_sha256
    if not pinned and not unfrozen:
        raise ValueError(f"{profile.name}: extension sha256 {extension_sha} is not the pinned {profile.extension_sha256}")
    return pinned


def merge_question_sources(frozen: list[dict], extension: list[dict]) -> list[dict]:
    """The bank's questions: the frozen set and the bank-only extension, each tagged with its
    ``question_source``. No id or base question id may occur in both (or twice in one), and an
    extension record carries no question text: it is never shown to an agent or a judge."""
    seen: set[str] = set()
    out: list[dict] = []
    for source, rows in (("frozen", frozen), ("extension", extension)):
        for q in rows:
            if source == "extension" and q.get("question") is not None:
                raise ValueError(f"extension record {q['id']} has a question text; extension records are bank-only")
            ids = {q["id"], q.get("base_question_id", q["id"])}
            clash = sorted(ids & seen) if source == "extension" else sorted({q["id"]} & seen)
            if clash:
                raise ValueError(f"question id(s) {', '.join(clash)} occur in more than one question record")
            seen |= ids
            out.append({**q, "question_source": source})
    return out


def exposed_path(graph: str, root: Path = ROOT) -> Path:
    """The graph's list of bank items exposed to compiler outputs (design ruling in the design log): the
    form that names each item's query when it exists, else the older list of ids."""
    d = root / "data" / "faults" / graph
    return d / "exposed-items.txt" if (d / "exposed-items.txt").exists() and not (d / "exposed-items.tsv").exists() \
        else d / "exposed-items.tsv"


def parse_exposed(text: str) -> list[str] | list[tuple[str, str]]:
    """An exposure list: one item id per line (the older form), or tab-separated ``id``, ``query_sha256``
    and provenance columns, which name the exact query the compiler's output was seen on."""
    rows = [line.split("\t") for line in text.splitlines() if line.strip() and not line.startswith("#")]
    if rows and all(len(r) >= 2 for r in rows):
        return [(r[0], r[1]) for r in rows]
    return [r[0].strip() for r in rows]


def mark_exposed(items: list[dict], listed: list[str] | list[tuple[str, str]]) -> dict:
    """Set ``exposed`` on every item and return the sensitivity stratum: exposed items leave the
    confirmatory population and the primary exploratory analyses, and are reported apart.

    An item is exposed when the compiler's output was seen on it (design ruling in the design log). A list
    that names the queries seen marks by query (``query_sha256``): after the question set changed, an
    id can name another question's item, and the same query can return under another id. Every item
    then also carries ``exposed_by_id``, the id reading, so the other reading needs no rebuild. A list
    of ids alone marks by id and adds no field (a bank built from one stays byte for byte as it was)."""
    pairs = [x for x in listed if isinstance(x, tuple)]
    ids = {x[0] if isinstance(x, tuple) else x for x in listed}
    shas = {sha for _, sha in pairs}
    by_query = bool(pairs)
    present_ids = {i["id"] for i in items}
    present_shas = {i.get("query_sha256") for i in items}
    by_class: dict[str, Counter] = defaultdict(Counter)
    by_split: Counter = Counter()
    for i in items:
        if by_query:
            i["exposed_by_id"] = i["id"] in ids
            i["exposed"] = i.get("query_sha256") in shas
        else:
            i["exposed"] = i["id"] in ids
        if i["exposed"]:
            by_split[i["split"]] += 1
            cls = (i["fault_class"] or "control") + (f"@{i['stratum']}" if i.get("stratum") else "")
            by_class[cls][i["bank"]] += 1
    out = {"rule": "query_sha256" if by_query else "id", "listed": len(listed),
           "in_bank": sum(1 for _, sha in pairs if sha in present_shas) if by_query else len(ids & present_ids),
           "in_bank_by_id": len(ids & present_ids),
           "missing": sorted(x[0] for x in pairs if x[1] not in present_shas) if by_query else sorted(ids - present_ids),
           "by_split": dict(sorted(by_split.items())),
           "by_class": {c: dict(sorted(n.items())) for c, n in sorted(by_class.items())}}
    if by_query:
        out["by_id_only"] = sorted(i["id"] for i in items if i["exposed_by_id"] and not i["exposed"])
        out["by_query_only"] = sorted(i["id"] for i in items if i["exposed"] and not i["exposed_by_id"])
    return out


def question_rules(rules: dict) -> ScoringRules:
    """Question files carry the rules' fields without a version (as the agent loop reads them). A bank is labelled under
    the v4 semantics whatever the scorer's defaults (``store.bank_rules``, change-log entry 120)."""
    return bank_rules(ScoringRules.from_json(rules) if "version" in rules else ScoringRules(**rules))


def _question_fields(q: dict) -> dict:
    # bank_only: no question text, so nothing that needs one (the judge's items) may take the item
    return {"question_id": q["id"], "base_question_id": q.get("base_question_id", q["id"]), "split": q["split"],
            "template": q["template"], "tier": q.get("tier"), "engine_sensitive": q.get("engine_sensitive", []),
            "question_source": q.get("question_source", "frozen"), "bank_only": not isinstance(q.get("question"), str)}


def gold_refs(q: ParsedQuery) -> list[str]:
    """The reference IRIs of a question's replacement forms: its gold constants, else every IRI of its
    gold (the constants check's rule)."""
    return sorted({s.iri for s in q.slots if s.role == "constant"}) or sorted({s.iri for s in q.slots})


def introduced_constants(gold: ParsedQuery, other: ParsedQuery) -> list[str]:
    """Constants of ``other`` that the gold does not have (a control's introduced constants)."""
    have = {s.iri for s in gold.slots if s.role == "constant"}
    return sorted({s.iri for s in other.slots if s.role == "constant"} - have)


def internal_node_of(slot: dict | None, replacement: str | None) -> bool:
    """A mutation's descriptor: its replacement is a constant and a store-internal node. A class or
    predicate replacement is no constant; an aggregation or a rejection replaces nothing."""
    return bool(slot and slot.get("role") == "constant" and replacement and internal(replacement))


def build_items(questions: list[dict], ctx: Context, runner: Runner, labellers: dict[str, Labeller],
                operators: tuple[str, ...] = OPERATORS, rewrites: RewriteSettings | None = None,
                rewrite_log: list | None = None, descriptive: tuple[str, ...] = DESCRIPTIVE) -> tuple[list[dict], list[dict]]:
    """All bank items (controls, mutations, rejections, accepted rewrites, descriptive strata) and the
    not-applicable skips; with ``rewrites``, each question's rejected rewrites and report go to
    ``rewrite_log``."""
    items: list[dict] = []
    skips: list[dict] = []
    for q in sorted(questions, key=lambda x: x["id"]):
        rules = question_rules(q["scoring_rules"])
        gold = runner.run(q["gold_sparql"])
        if gold.status != "ok":
            raise RuntimeError(f"gold query of {q['id']} failed on the store: {gold.error_kind}: {gold.error}")
        if gold.truncated:  # a partial gold is no baseline: comparing with it could mislabel the answer effect
            raise RuntimeError(f"gold query of {q['id']} was cut by the row cap; raise --row-cap")
        recorded = [row_from_json(r) for r in q["gold_rows"]]
        matches = score_rows(gold.rows or [], recorded, rules).exact
        base = _question_fields(q)
        parsed = parse(q["gold_sparql"], ctx.is_class)
        refs = gold_refs(parsed)
        items.append({"id": f"{q['id']}#control", "kind": "control", **base, "operator": "control",
                      "fault_class": None, "query": q["gold_sparql"], "query_sha256": query_sha(q["gold_sparql"]),
                      "answer_effect": {"label": "control", "n_rows": len(gold.rows or []),
                                        "gold_rerun_matches_recorded": matches},
                      "bank": "control", "control": "untouched_gold", "internal_node": False})  # introduces nothing
        if rewrites is not None:
            accepted, rejected, report = draw(q, gold, rules, rewrites)
            items.extend({**i, **base, "internal_node": any(internal(c) for c in introduced_constants(
                parsed, parse(i["query"], ctx.is_class)))} for i in accepted)
            if rewrite_log is not None:
                rewrite_log.append({"split": q["split"], "report": report, "rejected": rejected})
        for out in generate(q, ctx, operators):
            if isinstance(out, Skip):
                suffix = f"@d{out.distance}" if out.distance is not None else ""
                if out.rejected:
                    items.append({"id": f"{q['id']}#{out.operator}{suffix}", "kind": "mutation", **base,
                                  "operator": out.operator, "fault_class": FAULT_CLASS[out.operator],
                                  "distance": out.distance, "query": None,
                                  "answer_effect": {"label": "rejected", "reason": out.reason}, "bank": "rejected",
                                  "internal_node": False})
                else:
                    skips.append({"question_id": q["id"], "operator": out.operator, "distance": out.distance,
                                  "reason": out.reason})
                continue
            items.append(_mutation_item(q, base, out, gold, runner, rules, labellers, refs))
        if TYPE_CONSISTENT in descriptive:
            def effect(text: str, gold: Execution = gold, rules: ScoringRules = rules) -> dict:
                return answer_effect(gold, runner.run(text), rules)
            for out in type_consistent_swaps(q, parsed, ctx, effect):
                if isinstance(out, Skip):
                    skips.append({"question_id": q["id"], "operator": out.operator, "distance": None,
                                  "reason": out.reason, **(out.detail or {})})
                else:
                    items.append(_mutation_item(q, base, out, gold, runner, rules, labellers, refs))
    return items, skips


def realisable(runner: Runner, query: str | None) -> dict:
    """The descriptor "realisable in the data": the mutated pattern has at least one match."""
    if query is None:
        return {"realisable": None, "reason": "no_pattern"}
    ex = runner.run(query)
    if ex.status != "ok" or ex.boolean is None:
        return {"realisable": None, "reason": f"error:{ex.error_kind}"}
    return {"realisable": ex.boolean}


def _mutation_item(q: dict, base: dict, m: Mutation, gold: Execution, runner: Runner, rules: ScoringRules,
                   labellers: dict[str, Labeller], refs: list[str]) -> dict:
    effect = answer_effect(gold, runner.run(m.query), rules)
    labels = label_all(m.site, labellers)
    conformance = {name: lab["label"] for name, lab in labels.items()}
    details = {name: lab["violations"] for name, lab in labels.items() if lab["violations"]}
    # flags the gold query raises too (the difference reading): not the mutation's, kept for the audit
    discounted = {name: lab["discounted"] for name, lab in labels.items() if lab["discounted"]}
    bank = effect["label"]
    if m.operator in DESCRIPTIVE:
        if bank != "answer_changing":  # the operator keeps only answer-changing draws; the cached rerun must agree
            raise RuntimeError(f"{q['id']}: a kept {m.operator} draw is {bank} on rerun")
        bank = "descriptive"
        suffix = f"@c{m.selection['constant_index']}"
    else:
        suffix = f"@d{m.distance}" if m.distance is not None else ""
    item = {"id": f"{q['id']}#{m.operator}{suffix}", "kind": "mutation", **base,
            **{k: v for k, v in m.to_json().items() if k != "question_id"},
            "query_sha256": query_sha(m.query), "answer_effect": effect, "conformance": conformance,
            "conformance_violations": details, "conformance_discounted": discounted,
            "descriptors": realisable(runner, m.realisable_query), "bank": bank,
            "internal_node": internal_node_of(m.slot, m.replacement)}
    if m.slot and m.slot["role"] == "constant" and m.replacement:
        item["replacement_form"] = form(m.replacement, refs)
    if bank == "answer_neutral":
        item["control"] = "answer_neutral_mutation"  # its false-flag rate is reported apart from gold controls
        item["conformance_only"] = sorted(n for n, lab in conformance.items() if lab == "violates")
    return item


def class_key(i: dict) -> str:
    return i["fault_class"] + (f"@{i['stratum']}" if i.get("stratum") else "")


def in_w(i: dict) -> bool:
    """C4's W (and the pool of C2's type-wrong set): answer-changing mutations of the confirmatory
    operators, less items subsampled out by the wrong-item cap and items exposed to compiler outputs.
    A descriptive stratum has its own bank, so it is never in W."""
    return (i["kind"] == "mutation" and i["bank"] == "answer_changing" and not i.get("subsampled_out")
            and not i.get("exposed"))


ESTIMAND_RULE = ("W: answer-changing mutations (bank answer_changing) of the confirmatory operators, less items subsampled "
                 "out by the wrong-item cap and items exposed to compiler outputs. C2 type-wrong: the W items of class "
                 "type_wrong, also counted without internal-node items (the primary keeps them). C: untouched gold, "
                 "answer-neutral mutations and answer-preserving rewrites, less exposed items. A descriptive stratum "
                 "(bank descriptive: the type-consistent constant swap) is in none of them (design ruling (d), design log, "
                 "change-log entry 63).")
TC_KEYS = ("constants", "eligible", "answer_changing", "skipped_no_answer_change", "skipped_all_neutral",
           "skipped_with_rejections", "tries", "neutral_tries", "rejected_tries", "replacement_in_drawn_class",
           "replacement_covers_gold_classes", "questions_without_constant")


def _tries(tries: list[dict], c: Counter) -> tuple[int, int]:
    neutral = sum(t["outcome"] == "answer_neutral" for t in tries)
    rejected = sum(t["outcome"].startswith("rejected:") for t in tries)
    c["tries"] += len(tries)
    c["neutral_tries"] += neutral
    c["rejected_tries"] += rejected
    return neutral, rejected


def type_consistent_summary(items: list[dict], skips: list[dict], questions: dict[str, str]) -> dict:
    """Per split: the constants seen, the eligible ones (a class with another member: each is drawn
    from), the answer-changing items, the constants skipped after their tries (all neutral, or with
    rejections), the not-applicable reasons, the tries, and two checks of the replacement's own
    classes: it has the class drawn from (by construction: every item should), and it has every class
    of the gold constant (stricter: a Rhea compound's ChEBI class differs by design)."""
    tc: dict[str, Counter] = defaultdict(Counter)
    na: dict[str, Counter] = defaultdict(Counter)
    for i in items:
        if i["kind"] == "mutation" and i["operator"] in DESCRIPTIVE and i["bank"] == "descriptive":
            sel = i["selection"]
            for split in (i["split"], "all"):
                c = tc[split]
                c["constants"] += 1
                c["eligible"] += 1
                c["answer_changing"] += 1
                _tries(sel["tries"], c)
                c["replacement_in_drawn_class"] += sel["class"] in sel.get("replacement_classes", [])
                c["replacement_covers_gold_classes"] += set(sel["classes"]) <= set(sel.get("replacement_classes", []))
    for s in skips:
        if s["operator"] not in DESCRIPTIVE:
            continue
        for split in (questions.get(s["question_id"], "-"), "all"):
            c = tc[split]
            if s["reason"] == "no_constant_slot":
                c["questions_without_constant"] += 1
                continue
            c["constants"] += 1
            if s["reason"] == "no_answer_changing_draw":
                c["eligible"] += 1
                c["skipped_no_answer_change"] += 1
                neutral, rejected = _tries(s["tries"], c)
                c["skipped_all_neutral"] += neutral == len(s["tries"])
                c["skipped_with_rejections"] += rejected > 0
            else:
                na[split][s["reason"]] += 1
    return {split: {**{k: tc[split][k] for k in TC_KEYS}, "not_applicable": dict(sorted(na[split].items()))}
            for split in sorted(set(tc) | set(na))}


def estimand_sets(items: list[dict], descriptive_mix: dict) -> dict:
    splits = sorted({i["split"] for i in items}) + ["all"]
    w: dict[str, Counter] = {s: Counter() for s in splits}
    c2: dict[str, Counter] = {s: Counter() for s in splits}
    c2_bare: dict[str, Counter] = {s: Counter() for s in splits}
    q_all: dict[str, set] = {s: set() for s in splits}
    q_bare: dict[str, set] = {s: set() for s in splits}
    controls = {s: {"untouched_gold": 0, "answer_neutral": 0, "rewrites": 0} for s in splits}
    for i in items:
        for split in (i["split"], "all"):
            if in_w(i):
                w[split][class_key(i)] += 1
                if i["fault_class"] == "type_wrong":
                    c2[split][i.get("stratum") or "-"] += 1
                    q_all[split].add(i["question_id"])
                    if not i.get("internal_node"):
                        c2_bare[split][i.get("stratum") or "-"] += 1
                        q_bare[split].add(i["question_id"])
            elif not i.get("exposed"):
                if i["kind"] == "control":
                    controls[split]["untouched_gold"] += 1
                elif i["kind"] == "rewrite":
                    controls[split]["rewrites"] += 1
                elif i["kind"] == "mutation" and i["bank"] == "answer_neutral":
                    controls[split]["answer_neutral"] += 1
    return {"rule": ESTIMAND_RULE,
            "W": {s: dict(sorted(c.items())) for s, c in w.items()},
            "C2_type_wrong": {s: dict(sorted(c.items())) for s, c in c2.items()},
            "C2_type_wrong_without_internal": {s: dict(sorted(c.items())) for s, c in c2_bare.items()},
            "questions_with_type_wrong": {s: {"all": len(q_all[s]), "without_internal": len(q_bare[s])} for s in splits},
            "C": controls,
            "descriptive": {s: {cls: dict(sorted(c.items())) for cls, c in sorted(d.items())}
                            for s, d in sorted(descriptive_mix.items())}}


def summarise(items: list[dict], skips: list[dict], sources: list[str], rewrite_log: list | None = None,
              k: int | None = None, attempts: int | None = None) -> dict:
    questions = {i["question_id"]: i["split"] for i in items}
    by_op: dict[str, dict] = {op: {"answer_changing": 0, "answer_neutral": 0, "rejected": Counter(),
                                   "not_applicable": Counter()} for op in ALL_OPERATORS}
    for op in DESCRIPTIVE:
        by_op[op]["descriptive"] = True  # its answer-changing items are bank descriptive: outside W
    mix: dict[str, dict] = defaultdict(lambda: defaultdict(Counter))  # split -> class[@stratum] -> bank
    descriptive_mix: dict[str, dict] = defaultdict(lambda: defaultdict(Counter))  # split -> class@stratum -> effect
    strata: dict[str, Counter] = defaultdict(Counter)  # split -> question has both, near only, far only
    variants: Counter = Counter()
    conformance: dict[str, dict] = {s: defaultdict(Counter) for s in sources}
    # beside C2: its own set (in_w), with and without internal-node items (review of 508e601)
    conformance_c2: dict[str, dict] = {s: defaultdict(Counter) for s in sources}
    conformance_bare: dict[str, dict] = {s: defaultdict(Counter) for s in sources}
    conformance_descriptive: dict[str, dict] = {s: defaultdict(Counter) for s in sources}
    conformance_only: Counter = Counter()
    effect_basis: Counter = Counter()
    real: dict[str, Counter] = defaultdict(Counter)
    neutral_controls: Counter = Counter()
    swap_strata: dict[tuple[str, str], set] = defaultdict(set)
    internal_by_op: dict[str, Counter] = defaultdict(Counter)
    forms: dict[str, dict] = defaultdict(lambda: defaultdict(Counter))
    for i in items:
        if i["kind"] != "mutation":
            continue
        op, bank = i["operator"], i["bank"]
        if bank == "rejected":
            by_op[op]["rejected"][i["answer_effect"]["reason"]] += 1
            continue
        cls = class_key(i)
        key = f"{op}@{i['stratum']}" if i.get("stratum") else op
        internal_by_op[key]["items"] += 1
        internal_by_op[key]["internal"] += bool(i.get("internal_node"))
        if i.get("replacement_form"):
            for split in (i["split"], "all"):
                forms[key][split][i["replacement_form"]] += 1
        r = i.get("descriptors", {}).get("realisable")
        real[cls][{True: "realisable", False: "not_realisable", None: "not_assessed"}[r]] += 1
        if bank == "descriptive":
            label = i["answer_effect"]["label"]
            by_op[op][label] += 1
            for split in (i["split"], "all"):
                descriptive_mix[split][cls][label] += 1
            for s in sources:
                conformance_descriptive[s][cls][i["conformance"][s]] += 1
            continue
        by_op[op][bank] += 1
        effect_basis[i["answer_effect"].get("basis", "")] += 1
        for split in (i["split"], "all"):
            mix[split][cls][bank] += 1
            mix[split][i["fault_class"]][bank] += 1 if i.get("stratum") else 0
        if op in SWAPS:
            swap_strata[(i["question_id"], op)].add(i["stratum"])
        if i.get("variant"):
            variants[f"{op}:{i['variant']}:{bank}"] += 1
        if bank == "answer_changing":
            for s in sources:
                conformance[s][cls][i["conformance"][s]] += 1
                if i["fault_class"] == "type_wrong" and in_w(i):
                    conformance_c2[s][cls][i["conformance"][s]] += 1
                    if not i.get("internal_node"):
                        conformance_bare[s][cls][i["conformance"][s]] += 1
        elif bank == "answer_neutral":
            neutral_controls[f"{i['fault_class']}:{op}"] += 1
            for s in i.get("conformance_only", []):
                conformance_only[s] += 1
    for (qid, op), got in swap_strata.items():
        key = "both" if got == {"near", "far"} else f"{next(iter(got))}_only"
        for split in (questions[qid], "all"):
            strata[split][f"{op}:{key}"] += 1
    for s in skips:
        by_op[s["operator"]]["not_applicable"][s["reason"] + (f"@d{s['distance']}" if s.get("distance") else "")] += 1
    # a descriptive draw the store refused is a try, never an item: it is counted in "type_consistent", not
    # among the rejections (which count items), so the published rejection counts do not depend on (d)
    controls = [i for i in items if i["kind"] == "control"]
    rewrites = Counter(i["operator"] for i in items if i["kind"] == "rewrite")
    out = {
        "questions": {"total": len(questions), "dev": sum(1 for v in questions.values() if v == "dev"),
                      "test": sum(1 for v in questions.values() if v == "test")},
        "controls": {"untouched_gold": len(controls),
                     "answer_neutral_mutations": dict(sorted(neutral_controls.items())),
                     "answer_preserving_rewrites": dict(sorted(rewrites.items())),
                     # pooled: every correct query; the rewrites are also their own stratum (change-log entry 39)
                     "pooled": len(controls) + sum(neutral_controls.values()) + sum(rewrites.values()),
                     "gold_rerun_mismatch": sorted(i["question_id"] for i in controls
                                                   if not i["answer_effect"]["gold_rerun_matches_recorded"])},
        "by_operator": {op: {k: (dict(sorted(v.items())) if isinstance(v, Counter) else v) for k, v in d.items()}
                        for op, d in by_op.items()},
        "fault_mix": {split: {fc: dict(sorted(c.items())) for fc, c in sorted(d.items())} for split, d in sorted(mix.items())},
        "swap_strata_per_question": {split: dict(sorted(c.items())) for split, c in sorted(strata.items())},
        "realisable": {cls: dict(sorted(c.items())) for cls, c in sorted(real.items())},
        "variants": dict(sorted(variants.items())),
        "answer_effect_basis": dict(sorted(effect_basis.items())),
        "conformance_on_answer_changing": {s: {fc: dict(sorted(c.items())) for fc, c in sorted(d.items())}
                                           for s, d in conformance.items()},
        "conformance_on_type_wrong_in_w": {s: {fc: dict(sorted(c.items())) for fc, c in sorted(d.items())}
                                           for s, d in conformance_c2.items()},
        "conformance_on_type_wrong_without_internal": {s: {fc: dict(sorted(c.items())) for fc, c in sorted(d.items())}
                                                       for s, d in conformance_bare.items()},
        "conformance_on_descriptive": {s: {fc: dict(sorted(c.items())) for fc, c in sorted(d.items())}
                                       for s, d in conformance_descriptive.items()},
        "conformance_only": dict(sorted(conformance_only.items())),
        "rejections_total": sum(sum(d["rejected"].values()) for d in by_op.values()),
        "estimand_sets": estimand_sets(items, descriptive_mix),
        "internal_node": {"rule": "the constant an item introduces is a store-internal node (results/constants-check-"
                                  "20261019/shapes.py, harness/faults/shapes.py): a mutation's replacement when it "
                                  "replaces a constant; a control's constants not in the gold",
                          "by_operator": {k: {"items": c["items"], "internal": c["internal"]} for k, c in sorted(internal_by_op.items())},
                          "controls": {"untouched_gold": sum(bool(i.get("internal_node")) for i in controls),
                                       "rewrites": sum(bool(i.get("internal_node")) for i in items if i["kind"] == "rewrite")}},
        "replacement_forms": {k: {split: {f: c[f] for f in FORMS if c[f]} for split, c in sorted(d.items())}
                              for k, d in sorted(forms.items())},
        "type_consistent": type_consistent_summary(items, skips, questions),
    }
    if rewrite_log is not None:
        out["rewrites"] = rewrite_summary(rewrite_log, k or 0, attempts or 0)
    return out


def rewrite_summary(log: list[dict], k: int, attempts: int) -> dict:
    """Per question: accepted rewrites (exact and at least n), and per operator: candidates, tries,
    acceptances, rejections by reason, untried candidates and why a question had none."""
    exact: dict[str, Counter] = defaultdict(Counter)
    ops = {op: {"candidates": 0, "tried": 0, "accepted": 0, "rejected": Counter(), "untried": 0, "no_candidate": Counter()}
           for op in REWRITES}
    stop: Counter = Counter()
    for entry in log:
        r = entry["report"]
        for split in (entry["split"], "all"):
            exact[split][r["accepted"]] += 1
        stop[r["stop"]] += 1
        for op in REWRITES:
            ops[op]["candidates"] += r["candidates"][op]
            ops[op]["untried"] += r["untried"][op]
            ops[op]["accepted"] += r["accepted_by_operator"][op]
            ops[op]["tried"] += r["accepted_by_operator"][op]
            if op in r["reasons"]:
                ops[op]["no_candidate"][r["reasons"][op]] += 1
        for rej in entry["rejected"]:
            ops[rej["operator"]]["rejected"][rej["reason"]] += 1
            ops[rej["operator"]]["tried"] += rej["reason"] != "duplicate"
    return {"k": k, "attempt_cap": attempts,
            "accepted_per_question": {s: {str(n): c[n] for n in range(k + 1)} for s, c in sorted(exact.items())},
            "reaching_at_least": {s: {str(n): sum(v for m, v in c.items() if m >= n) for n in range(1, k + 1)}
                                  for s, c in sorted(exact.items())},
            "by_operator": {op: {key: (dict(sorted(v.items())) if isinstance(v, Counter) else v) for key, v in d.items()}
                            for op, d in ops.items()},
            "stop": dict(sorted(stop.items()))}


def cap_wrong(items: list[dict], cap: int, seed: int) -> dict:
    """Cap the wrong (answer-changing) items per question at ``cap`` by a seeded subsample; the
    others stay in the bank marked ``subsampled_out``. Reads no label."""
    import hashlib as _h
    import random

    wrong: dict[str, list[dict]] = defaultdict(list)
    for i in items:
        if i["kind"] == "mutation" and i["bank"] == "answer_changing":
            i["subsampled_out"] = False
            wrong[i["question_id"]].append(i)
    by_class: Counter = Counter()
    capped = 0
    for qid, group in sorted(wrong.items()):
        if len(group) <= cap:
            continue
        capped += 1
        rng = random.Random(int(_h.sha256(f"{seed}|{qid}|cap-wrong".encode()).hexdigest()[:16], 16))
        keep = set(rng.sample(sorted(i["id"] for i in group), cap))
        for i in group:
            if i["id"] not in keep:
                i["subsampled_out"] = True
                by_class[i["fault_class"] + (f"@{i['stratum']}" if i.get("stratum") else "")] += 1
    return {"items": items, "summary": {"cap": cap, "questions_capped": capped, "subsampled_out": sum(by_class.values()),
                                        "by_class": dict(sorted(by_class.items()))}}


class FrozenBankChanged(RuntimeError):
    """A rebuild into a frozen bank's directory that would change a data file or an input."""


@dataclass(frozen=True)
class BankWrite:
    manifest: dict
    verified: bool  # True: the directory held a frozen bank, and this build reproduced it (nothing written)


FREEZE_RULE = ("frozen before any checker runs on the bank (design log; design log, criterion 5): every input pinned and "
               "recorded by sha256, the store identified live, the data files hashed; a rebuild into this directory is "
               "refused if it would change any of them")


def freeze_record(header: dict, hashes: dict[str, str]) -> dict:
    """The freeze: every input present with its sha256, the absent ones named, the store and the data
    files. Only a final build may freeze: not provisional, its questions the graph's frozen set, its
    extension (if any) the pinned one."""
    inputs = header["inputs"]
    if header.get("provisional"):
        raise ValueError("a provisional bank cannot be frozen")
    if not inputs["questions"].get("frozen"):
        raise ValueError("the question file is not the graph's frozen set: the bank cannot be frozen")
    if inputs.get("extension") is not None and not inputs["extension"].get("pinned"):
        raise ValueError("the extension is not the graph's pinned one: the bank cannot be frozen")
    shas: dict[str, str] = {}
    absent: list[str] = []
    for name, entry in sorted(inputs.items()):
        if entry is None or entry == {}:
            absent.append(name)
            continue
        group = entry if name == "sources" else {None: entry}
        for sub, e in group.items():
            if not e.get("sha256"):
                raise ValueError(f"input {name}{'.' + sub if sub else ''} has no sha256: the bank cannot be frozen")
            shas[f"{name}.{sub}" if sub else name] = e["sha256"]
    return {"rule": FREEZE_RULE, "inputs_sha256": shas, "absent_inputs": absent,
            "store": header["store"]["identity"]["key"], "files_sha256": dict(hashes)}


def _frozen(out_dir: Path) -> dict | None:
    path = out_dir / "manifest.json"
    if not path.exists():
        return None
    m = json.loads(path.read_text())
    return m if m.get("frozen") else None


def _changed_items(out_dir: Path, text: str) -> list[str]:
    """The ids of items that differ from a frozen bank's (items.jsonl, or the committed items.jsonl.gz)."""
    old_path = out_dir / "items.jsonl"
    if old_path.exists():
        old_text = old_path.read_text()
    elif (out_dir / "items.jsonl.gz").exists():
        with gzip.open(out_dir / "items.jsonl.gz", "rt") as fh:
            old_text = fh.read()
    else:
        return []
    old = {json.loads(line)["id"]: line for line in old_text.splitlines()}
    new = {json.loads(line)["id"]: line for line in text.splitlines()}
    return sorted(i for i in set(old) | set(new) if old.get(i) != new.get(i))


def _check_frozen(frozen: dict, out_dir: Path, header: dict, texts: dict[str, str], hashes: dict[str, str]) -> None:
    problems: list[str] = []
    for name in sorted(set(frozen["files"]) | set(hashes)):
        if frozen["files"].get(name) != hashes.get(name):
            problems.append(f"file {name}")
            if name == "items.jsonl":
                ids = _changed_items(out_dir, texts.get(name, ""))
                problems[-1] += f" ({len(ids)} items differ: {', '.join(ids[:10])}{', ...' if len(ids) > 10 else ''})"
    for name in sorted(set(frozen["inputs"]) | set(header["inputs"])):
        a, b = frozen["inputs"].get(name), header["inputs"].get(name)
        if json.dumps(_shas(a), sort_keys=True) != json.dumps(_shas(b), sort_keys=True):
            problems.append(f"input {name}")
    if problems:
        raise FrozenBankChanged(f"{out_dir} holds a frozen bank; this build would change: " + "; ".join(problems))


def _shas(entry):
    """An input entry reduced to what identifies it: its sha256 (per source for the sources)."""
    if not entry:
        return None
    if "sha256" in entry:
        return entry["sha256"]
    return {k: (v or {}).get("sha256") for k, v in sorted(entry.items())}


def write_bank(out_dir: Path, header: dict, items: list[dict], skips: list[dict],
               extra: dict[str, list[dict]] | None = None, *, freeze: bool = False) -> BankWrite:
    files = {"items.jsonl": items, "not-applicable.jsonl": skips, **(extra or {})}
    texts = {name: "".join(json.dumps(r, sort_keys=True) + "\n" for r in records) for name, records in files.items()}
    hashes = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in texts.items()}
    frozen = _frozen(out_dir)
    if frozen is not None:  # a frozen bank is only ever checked, never rewritten
        _check_frozen(frozen, out_dir, header, texts, hashes)
        return BankWrite(frozen, True)
    record = freeze_record(header, hashes) if freeze else None  # refused before anything is written
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in texts.items():
        (out_dir / name).write_text(text)
    manifest = {**header, "schema": SCHEMA, "frozen": freeze, "files": hashes, **({"freeze": record} if record else {})}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return BankWrite(manifest, False)


ROWS = (("type_wrong", "near"), ("type_wrong", "far"), ("vocabulary_absent", None), ("stale_in_data", None),
        ("meaning", None))


def tables_markdown(items: list[dict], summary: dict, sources: list[str], skips: list[dict],
                    agents: dict | None = None) -> str:
    """Post-able tables: the class mix by split and distance, operators, controls, the realisable
    descriptor and conformance per source (answer-changing mutations, the common denominator)."""
    muts = [i for i in items if i["kind"] == "mutation" and i["bank"] != "rejected"]

    def count(cls, stratum, **where):
        return sum(1 for i in muts if i["fault_class"] == cls and i.get("stratum") == stratum
                   and all(i.get(k) == v for k, v in where.items()))

    def name(cls, stratum):
        if stratum in ("near", "far"):
            return f"{cls} ({stratum}, d = {2 if stratum == 'near' else 4})"
        return f"{cls} ({stratum})" if stratum else cls

    out = ["### Class mix by split and distance (answer-changing / answer-neutral)", "",
           "| class | test | dev | all |", "|---|---|---|---|"]
    for cls, stratum in ROWS:
        cells = []
        for split in ("test", "dev", None):
            w = {"split": split} if split else {}
            cells.append(f"{count(cls, stratum, bank='answer_changing', **w)} / {count(cls, stratum, bank='answer_neutral', **w)}")
        out.append(f"| {name(cls, stratum)} | " + " | ".join(cells) + " |")
    tot = [f"{sum(1 for i in muts if i['bank'] == 'answer_changing' and (not s or i['split'] == s))} / "
           f"{sum(1 for i in muts if i['bank'] == 'answer_neutral' and (not s or i['split'] == s))}" for s in ("test", "dev", None)]
    out += [f"| all | {' | '.join(tot)} |", "",
            "### Swaps per stratum: drawn and missing (questions)", "",
            "| operator | near drawn | near missing | far drawn | far missing |", "|---|---|---|---|---|"]
    for op in ("swap_predicate", "swap_class", "swap_constant"):
        cells = []
        for d in (2, 4):
            drawn = sum(1 for i in items if i["kind"] == "mutation" and i["operator"] == op and i.get("distance") == d)
            missing = Counter(s["reason"] for s in skips if s["operator"] == op and s.get("distance") == d)
            cells += [str(drawn), f"{sum(missing.values())} (" + ", ".join(f"{k} {v}" for k, v in sorted(missing.items())) + ")"]
        out.append(f"| {op} | " + " | ".join(cells) + " |")
    out += ["", "### Operators", "", "| operator | answer-changing | answer-neutral | rejected | not applicable |",
            "|---|---|---|---|---|"]
    for op, x in summary["by_operator"].items():
        rej = ", ".join(f"{k} {v}" for k, v in x["rejected"].items()) or "0"
        na = ", ".join(f"{k} {v}" for k, v in x["not_applicable"].items()) or "0"
        label = f"{op} (descriptive; its refused draws are tries, below)" if x.get("descriptive") else op
        out.append(f"| {label} | {x['answer_changing']} | {x['answer_neutral']} | {rej} | {na} |")
    out += _internal_tables(items, summary, agents) + _descriptive_tables(summary)
    c = summary["controls"]
    out += ["", "### Controls (false-flag denominators, reported separately)", "", "| control | n |", "|---|---|",
            f"| untouched gold | {c['untouched_gold']} |"]
    out += [f"| answer-neutral {k} | {v} |" for k, v in c["answer_neutral_mutations"].items()]
    out += ["", "### Realisable in the data (descriptor; answer-changing mutations)", "",
            "| class | realisable | not realisable | not assessed |", "|---|---|---|---|"]
    for cls, stratum in ROWS:
        r = Counter({True: "realisable", False: "not_realisable", None: "not_assessed"}[i.get("descriptors", {}).get("realisable")]
                    for i in muts if i["fault_class"] == cls and i.get("stratum") == stratum and i["bank"] == "answer_changing")
        out.append(f"| {name(cls, stratum)} | {r['realisable']} | {r['not_realisable']} | {r['not_assessed']} |")
    labels = ("violates", "conforms", "undetermined", "no_axiom", "not_applicable")
    for s in sources:
        out += ["", f"### Conformance labels, {s} (answer-changing mutations)", "",
                "| class | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
        for cls, stratum in ROWS:
            r = Counter(i["conformance"][s] for i in muts if i["fault_class"] == cls and i.get("stratum") == stratum
                        and i["bank"] == "answer_changing")
            out.append(f"| {name(cls, stratum)} | " + " | ".join(str(r[x]) for x in labels) + " |")
        for cls, stratum in ROWS[:2]:  # C2's own set (in_w), so these rows share its denominators
            c2 = [i for i in muts if i["fault_class"] == cls and i.get("stratum") == stratum and in_w(i)]
            for suffix, sel in (("C2 set", c2), ("C2 set without internal-node items", [i for i in c2 if not i.get("internal_node")])):
                r = Counter(i["conformance"][s] for i in sel)
                out.append(f"| {name(cls, stratum)}, {suffix} | " + " | ".join(str(r[x]) for x in labels) + " |")
        r = Counter(i["conformance"][s] for i in muts if i["bank"] == "descriptive")
        out.append("| meaning (type_consistent, descriptive) | " + " | ".join(str(r[x]) for x in labels) + " |")
    return "\n".join(out) + "\n"


def _internal_tables(items: list[dict], summary: dict, agents: dict | None) -> list[str]:
    """C2 (ii)'s and the type-wrong rates' denominators with and without internal-node items, the
    descriptor per operator, and the replacement forms of constants."""
    out = ["", ("### Type-wrong items with and without internal-node items (W; n (questions): all / without; "
                "the primary keeps them)"), "", "| class | test | dev | all |", "|---|---|---|---|"]
    for stratum in ("near", "far", None):
        cells = []
        for split in ("test", "dev", None):
            sel = [i for i in items if in_w(i) and i["fault_class"] == "type_wrong" and (stratum is None or i.get("stratum") == stratum)
                   and (split is None or i["split"] == split)]
            bare = [i for i in sel if not i.get("internal_node")]
            cells.append(f"{len(sel)} ({len({i['question_id'] for i in sel})}) / "
                         f"{len(bare)} ({len({i['question_id'] for i in bare})})")
        label = f"type_wrong ({stratum}, d = {2 if stratum == 'near' else 4})" if stratum else "type_wrong (both strata)"
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    out += ["", ("### Internal-node items per operator (non-rejected mutations whose replacement constant is a "
                 "store-internal node)"), "", "| operator | items | internal node |", "|---|---|---|"]
    out += [f"| {k} | {c['items']} | {c['internal']} |" for k, c in summary["internal_node"]["by_operator"].items()]
    out += ["", "### Replacement forms of constants (against the question's gold constants; the constants check's rule)", "",
            "| operator | split | n | " + " | ".join(FORMS) + " |", "|---|---|---|" + "---|" * len(FORMS)]
    for k, by_split in summary["replacement_forms"].items():
        for split in ("test", "dev", "all"):
            c = by_split.get(split, {})
            out.append(f"| {k} | {split} | {sum(c.values())} | " + " | ".join(str(c.get(f, 0)) for f in FORMS) + " |")
    return out + _agent_forms(summary, agents)


def _agent_forms(summary: dict, agents: dict | None) -> list[str]:
    """The bank's replacement forms (all splits) beside the agents' wrong constants on the same graph,
    as shares of n (``agents``: the graph's entry of an agent-forms file, or None when there is none)."""
    out = ["", "### Replacement forms beside agents' wrong constants (% of n; all splits)", ""]
    if not agents:
        return out + ["Not measured: the constants check read agents' wrong constants on the UniProt calibration rounds only."]
    out += [f"Agents: {agents.get('source', '')}", "", "| population | n | " + " | ".join(FORMS) + " |",
            "|---|---|" + "---|" * len(FORMS)]

    def row(label: str, c: dict) -> str:
        n = sum(c.get(f, 0) for f in FORMS)
        return f"| {label} | {n} | " + " | ".join(f"{c.get(f, 0)} ({100 * c.get(f, 0) / n:.0f}%)" if n else "0"
                                                    for f in FORMS) + " |"
    out += [row(f"bank {k}", d.get("all", {})) for k, d in summary["replacement_forms"].items()]
    out += [row(f"agents {run}", c) for run, c in agents["runs"].items()]
    return out


def _descriptive_tables(summary: dict) -> list[str]:
    out = ["", "### Descriptive stratum: type-consistent constant swaps (meaning; outside C2 and C4's W)", "",
           ("| split | constants | eligible (drawn) | answer-changing | skipped: all draws neutral | skipped: with "
            "rejections | not applicable | tries | neutral tries | rejected tries | replacement in the drawn class | "
            "replacement has every class of the gold |"),
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for split, c in summary["type_consistent"].items():
        na = ", ".join(f"{k} {v}" for k, v in c["not_applicable"].items()) or "0"
        out.append(f"| {split} | {c['constants']} | {c['eligible']} | {c['answer_changing']} | {c['skipped_all_neutral']} | "
                   f"{c['skipped_with_rejections']} | {na} | {c['tries']} | {c['neutral_tries']} | {c['rejected_tries']} | "
                   f"{c['replacement_in_drawn_class']} | {c['replacement_covers_gold_classes']} |")
    e = summary["estimand_sets"]
    out += ["", "### Estimand sets (the descriptive stratum is in none)", "",
            ("| split | W | W by class | C2 type-wrong | C2 type-wrong without internal nodes | C (gold / neutral / "
             "rewrites) | descriptive |"), "|---|---|---|---|---|---|---|"]

    def fmt(d: dict) -> str:
        return ", ".join(f"{k} {v}" for k, v in d.items()) or "0"
    for split in e["W"]:
        w, ctl = e["W"][split], e["C"][split]
        desc = {cls: sum(v.values()) for cls, v in e["descriptive"].get(split, {}).items()}
        out.append(f"| {split} | {sum(w.values())} | {fmt(w)} | {fmt(e['C2_type_wrong'][split])} | "
                   f"{fmt(e['C2_type_wrong_without_internal'][split])} | {ctl['untouched_gold']} / {ctl['answer_neutral']} / "
                   f"{ctl['rewrites']} | {fmt(desc)} |")
    return out
