"""System prompts for the agent, one per prompt condition.

Each condition has a versioned id. The text a model receives is the template
rendered with the SPARQL tool's row cap and time-out, so a run record stores
the hash of the rendered text (what the model saw) next to the hash of the
template (what the condition is). Changing either string means a new id.

In the ``none`` condition the model gets the tools and the question, and no
text about the ontology or the graph's schema. The three ontology conditions
(``curated``, ``census``, ``generated``; design log) are the none text, byte for
byte, then the ontology block: a blank line, one framing sentence, a blank line,
and one source rendered by ``harness.sources.render`` at the graph's budget
(``scripts/build_prompts.py``). The block is therefore exactly the bytes an
ontology prompt adds to none-v1, and its hash is what a run records. The
framing is the same for every source and never says where the schema came
from; the tools, the task text and the answer format are those of ``none``.
The block lives in a content-addressed store (``BLOCK_DIR``) and a run names it
by its sha256 (``AgentConfig.ontology_block_sha256``), so a replay loads the
exact block it was run with, and a block is never overwritten.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

NONE_TEMPLATE = (
    "You answer a question by querying a knowledge graph through a SPARQL 1.1 endpoint. "
    "You have three tools. "
    "sparql_query runs a read-only SPARQL query and returns at most {row_cap} rows; "
    "queries time out after {timeout_s} seconds. "
    "submit_report submits your final answer as rows of RDF terms: one row per answer, "
    "with the columns in the order the question asks for. "
    "Write IRIs in full, not as prefixed names. "
    "Give numbers as numeric literals. "
    "abstain ends the attempt without an answer. "
    "Only the rows you submit are scored. "
    "Submitting or abstaining ends the attempt."
)

# The ontology conditions: the none text, then the block, which opens with a blank line, this sentence and a
# blank line before the rendered source. The block is appended after formatting, so a brace in it is text.
ONTOLOGY_FRAMING = "A description of this graph's schema follows."
BLOCK_PREFIX = "\n\n" + ONTOLOGY_FRAMING + "\n\n"
BLOCK_DIR = Path(__file__).resolve().parents[2] / "data" / "prompts" / "blocks"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
# Further block stores, searched after BLOCK_DIR: a private graph's blocks (a15) stay outside the repository, in the
# store its prompt manifest names (scripts/run_pilot.py ontology_block). Registered in this process and, through the
# environment variable, in the processes it starts (a replay, a checker); a block is still found only by its sha256
# and refused unless its bytes have it, so where it was found changes nothing a record holds.
BLOCK_STORES_ENV = "WWW2027_BLOCK_STORES"
_EXTRA_STORES: list[Path] = []

# Sent as a user message when the model answers in text without calling a tool.
NUDGE_TEXT = "Call submit_report or abstain."


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PromptCondition:
    condition: str
    id: str
    template: str
    nudge: str = NUDGE_TEXT
    ontology: bool = False  # the template is followed by an ontology block

    @property
    def template_sha256(self) -> str:
        return sha256_text(self.template)

    def render(self, *, row_cap: int, timeout_s: float, ontology_block: str | None = None) -> str:
        if not isinstance(row_cap, int) or isinstance(row_cap, bool) or row_cap <= 0:
            raise ValueError(f"the agent's SPARQL tool needs a positive row cap, got {row_cap!r}")
        text = self.template.format(row_cap=row_cap, timeout_s=f"{timeout_s:g}")
        if not self.ontology:
            if ontology_block is not None:
                raise ValueError(f"condition {self.condition!r} takes no ontology block")
            return text
        if not ontology_block or not ontology_block.strip():
            raise ValueError(f"condition {self.condition!r} needs an ontology block")
        if not ontology_block.startswith(BLOCK_PREFIX) or not ontology_block[len(BLOCK_PREFIX):].strip():
            raise ValueError("an ontology block is the fixed framing then a rendered source (ontology_block())")
        return text + ontology_block

    def record(self, rendered: str, ontology_block: str | None = None) -> dict:
        """What a run record stores about the prompt the model was given."""
        out = {
            "condition": self.condition,
            "id": self.id,
            "sha256": sha256_text(rendered),
            "template_sha256": self.template_sha256,
            "nudge": self.nudge,
            "nudge_sha256": sha256_text(self.nudge),
        }
        if self.ontology:
            if ontology_block is None:
                raise ValueError(f"condition {self.condition!r} records its ontology block's hash")
            out["ontology_block_sha256"] = sha256_text(ontology_block)
        return out


CONDITIONS: dict[str, PromptCondition] = {
    "none": PromptCondition(condition="none", id="none-v1", template=NONE_TEMPLATE),
    # aligned-curated: the curated source with a reviewed mapping applied (a15, H5; design ruling in the design log,
    # 20 Oct 12:54), on the same template as every ontology condition.
    **{c: PromptCondition(condition=c, id=f"{c}-v1", template=NONE_TEMPLATE, ontology=True)
       for c in ("curated", "census", "generated", "aligned-curated")},
}


def get_condition(condition: str) -> PromptCondition:
    try:
        return CONDITIONS[condition]
    except KeyError:
        raise ValueError(f"unknown prompt condition {condition!r}; known: {sorted(CONDITIONS)}") from None


def ontology_block(rendered: str) -> str:
    """The block an ontology prompt adds to none-v1: the fixed framing, then the rendered source."""
    if not rendered or not rendered.strip():
        raise ValueError("an ontology block needs a rendered source")
    return BLOCK_PREFIX + rendered


def _block_path(sha256: str, root: Path | None) -> Path:
    if not isinstance(sha256, str) or not _SHA256.match(sha256):
        raise ValueError(f"an ontology block is named by its lowercase sha256, got {sha256!r}")
    return (root or BLOCK_DIR) / f"{sha256}.txt"


def add_block_store(path: Path) -> None:
    """Search ``path`` for blocks after BLOCK_DIR, here and in child processes (``BLOCK_STORES_ENV``)."""
    path = Path(path).resolve()
    if path not in _EXTRA_STORES:
        _EXTRA_STORES.append(path)
    listed = [s for s in os.environ.get(BLOCK_STORES_ENV, "").split(os.pathsep) if s]
    if str(path) not in listed:
        os.environ[BLOCK_STORES_ENV] = os.pathsep.join([*listed, str(path)])


def _stores() -> list[Path]:
    env = [Path(s) for s in os.environ.get(BLOCK_STORES_ENV, "").split(os.pathsep) if s]
    return [BLOCK_DIR, *_EXTRA_STORES, *(s for s in env if s not in _EXTRA_STORES)]


def load_block(sha256: str, root: Path | None = None) -> str:
    """The stored block with this hash, from ``root``, else from BLOCK_DIR or a registered store (``add_block_store``);
    a block whose bytes no longer match its name is refused."""
    path = _block_path(sha256, root)
    if root is None and not path.exists():
        path = next((s / path.name for s in _stores() if (s / path.name).exists()), path)
    text = path.read_bytes().decode("utf-8")
    if sha256_text(text) != sha256:
        raise ValueError(f"ontology block {sha256[:16]}: its content does not match its sha256")
    return text


def store_block(text: str, root: Path | None = None) -> str:
    """Add a block to the store (append-only) and return its sha256."""
    sha = sha256_text(text)
    path = _block_path(sha, root)
    if path.exists():
        load_block(sha, root)  # an existing block is checked, never overwritten
        return sha
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return sha


def user_message(question: dict) -> str:
    """The user turn: the question text and nothing else."""
    for key in ("text", "question"):
        value = question.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError("question has no 'text' (or 'question') string")
