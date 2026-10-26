"""The two closed-book probes' system prompts (design log, E1.7; preregistration §6).

Each probe's system prompt is the ``none-v1`` template minus the sentences that
describe the tools, plus exactly one added instruction:

* probe (a), ``closed-book-answer-v2``: answer from memory, a best guess when
  unsure, with the rows only, in the JSON row/cell format ``submit_report`` accepts,
  which the instruction spells out (parse.py reads it). Design ruling in the design log:
  (a) also drops the template's first sentence ("You answer a question by querying a
  knowledge graph through a SPARQL 1.1 endpoint."), which invites "I cannot query"
  replies, and keeps "Only the rows you submit are scored.";
* probe (b), ``closed-book-query-v1``: "Write one SPARQL query that answers this
  question; you cannot run it." (parse.py extracts the query, run.py executes it). It
  keeps the first sentence: it frames the task and names the query language.

What is removed is fixed by ``TOOL_SENTENCES``: five whole sentences of the
template, each naming a tool or the end of an attempt by a tool (and, for probe (a),
``FIRST_SENTENCE``). The bases keep the other sentences verbatim and in order. With
the tool sentences go the row cap and time-out placeholders, so a probe's system
prompt is the same text on every graph and is pinned by its sha256 below. Changing
any of these strings means a new probe id; a retired id stays here (``RETIRED``),
pinned, so its records still replay.
"""

from __future__ import annotations

from dataclasses import dataclass

from harness.agent.prompts import CONDITIONS, NONE_TEMPLATE, sha256_text

BASE_CONDITION = CONDITIONS["none"]  # none-v1

# The sentences of NONE_TEMPLATE that describe the tools, removed for the probes.
TOOL_SENTENCES = (
    "You have three tools.",
    "sparql_query runs a read-only SPARQL query and returns at most {row_cap} rows; "
    "queries time out after {timeout_s} seconds.",
    "submit_report submits your final answer as rows of RDF terms: one row per answer, "
    "with the columns in the order the question asks for.",
    "abstain ends the attempt without an answer.",
    "Submitting or abstaining ends the attempt.",
)


FIRST_SENTENCE = "You answer a question by querying a knowledge graph through a SPARQL 1.1 endpoint."


def base_text(also_removed: tuple[str, ...] = ()) -> str:
    """NONE_TEMPLATE with TOOL_SENTENCES (and ``also_removed``) deleted; the template joins its
    sentences with one space."""
    text = NONE_TEMPLATE
    for sentence in (*TOOL_SENTENCES, *also_removed):
        if sentence not in text:
            raise ValueError(f"none-v1 no longer holds the sentence {sentence[:40]!r}")
        text = text.replace(sentence, "")
    return " ".join(text.split())


BASE_TEXT = base_text()  # probe (b); probe (a) v1
ANSWER_BASE_TEXT = base_text((FIRST_SENTENCE,))  # probe (a) from v2

ROWS_FORMAT = (
    '{"rows": [[cell, ...], ...]}, one row per answer with the columns in the order the question asks for, '
    'where each cell is an IRI {"type": "iri", "value": ...}, a literal '
    '{"type": "literal", "value": ..., "datatype"?: ..., "lang"?: ...}, a JSON number or boolean, or null.'
)
ANSWER_INSTRUCTION = (
    "You cannot query anything. Answer from what you know, as rows only. If unsure, give your best guess. "
    f"Reply with one JSON object and nothing else, {ROWS_FORMAT}"
)
ANSWER_INSTRUCTION_V1 = f"Answer with the rows only: reply with one JSON object and nothing else, {ROWS_FORMAT}"
QUERY_INSTRUCTION = "Write one SPARQL query that answers this question; you cannot run it."


@dataclass(frozen=True)
class ProbeSpec:
    probe: str  # "answer" (a) or "query" (b)
    id: str
    instruction: str
    base: str = BASE_TEXT

    @property
    def system(self) -> str:
        return f"{self.base} {self.instruction}"

    @property
    def system_sha256(self) -> str:
        return sha256_text(self.system)

    @property
    def instruction_sha256(self) -> str:
        return sha256_text(self.instruction)

    def record(self) -> dict:
        """What a probe record stores about the prompt."""
        return {
            "probe": self.probe,
            "id": self.id,
            "base_condition": BASE_CONDITION.id,
            "base_template_sha256": BASE_CONDITION.template_sha256,
            "base_sha256": sha256_text(self.base),
            "instruction_sha256": self.instruction_sha256,
            "sha256": self.system_sha256,
            "tools": [],
        }


PROBES: dict[str, ProbeSpec] = {  # what a run uses
    "answer": ProbeSpec("answer", "closed-book-answer-v2", ANSWER_INSTRUCTION, ANSWER_BASE_TEXT),
    "query": ProbeSpec("query", "closed-book-query-v1", QUERY_INSTRUCTION),
}
RETIRED: tuple[ProbeSpec, ...] = (  # never run (superseded before any call), kept pinned for replay
    ProbeSpec("answer", "closed-book-answer-v1", ANSWER_INSTRUCTION_V1),
)
PROBE_IDS = {spec.id: spec for spec in (*PROBES.values(), *RETIRED)}


def get_probe(probe: str) -> ProbeSpec:
    """A probe by name ("answer", "query") or id ("closed-book-answer-v1", ...)."""
    spec = PROBES.get(probe) or PROBE_IDS.get(probe)
    if spec is None:
        raise ValueError(f"unknown probe {probe!r}; known: {sorted(PROBES)} or {sorted(PROBE_IDS)}")
    return spec
