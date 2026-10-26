"""What a question's answer is scored against when a plain table of gold rows cannot say it (issue 10 rulings).

A question record may carry an optional ``gold_rule`` next to its ``gold_rows``; without one, the gold rows are the
gold, exactly as before. The rule's JSON form is the per-question entry of a set's ``scoring.json``; extra keys
(query hash, columns, order, offset, limit, notes) are provenance and are ignored here.

``tie_aware``          An ORDER BY ... LIMIT window whose cut falls inside a tie. ``required``: the window's rows whose
                       ordering value no row outside the window shares. ``tied_groups``: for each value shared across
                       a cut, ``slots`` (the window positions it fills) and ``members`` (every row with that value).
                       Every valid fill, the required rows plus ``slots`` members of each group, is a plain gold
                       table of len(required) + sum(slots) rows, and the answer's score is the best plain score over
                       the fills (``scorer.py``; at most ``MAX_TIE_FILLS`` fills).
``either_value``       ``alternatives``: an object of named row tables; the answer is scored against each as a whole,
                       under the question's ordinary rules, and the best alternative counts.
``numeric_tolerance``  A mark only: the gold rows are scored as usual and the ruling rests on the rules' numeric
                       tolerance, so ``question_gold`` refuses a question whose rules have none.

``anchor_names`` (scorer-rules-v5, change-log entry 120) is not a rule of the question record but a form
``question_gold`` builds around the question's gold (its rows, or its tie-aware or either-value rule): ``gold``, that
gold, and ``anchors``, the names by which the question names its anchor entities, each with its IRI
(``question_anchor_names``). Under the rules' ``anchor_names`` the scorer reads an untagged answer literal equal to one
of these names as its IRI, in a column whose bound gold cells are all IRIs, in every table of the gold; with the rule
off, or with no such name, the gold scores exactly as it does alone.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from harness.terms import IRI, BNode, Literal, Row, term_from_json

from .rules import ScoringRules


def _rows(rows: Sequence[Sequence[Any]]) -> list[Row]:
    """Rows of term JSON (or of terms) -> rows of terms, as the agent loop reads gold rows."""
    return [tuple(c if isinstance(c, (IRI, Literal, BNode)) else term_from_json(c) for c in row) for row in rows]


def _int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class TiedGroup:
    """Rows sharing one ordering value across a cut of the window; ``slots`` of them belong in the answer."""

    slots: int
    members: tuple[Row, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "members", tuple(tuple(r) for r in self.members))
        if not _int(self.slots) or not 1 <= self.slots <= len(self.members):
            raise ValueError(f"a tied group needs 1 <= slots <= {len(self.members)} members, got slots {self.slots!r}")


@dataclass(frozen=True)
class TieAwareGold:
    required: tuple[Row, ...]
    groups: tuple[TiedGroup, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "required", tuple(tuple(r) for r in self.required))
        object.__setattr__(self, "groups", tuple(self.groups))
        if not all(isinstance(g, TiedGroup) for g in self.groups):
            raise TypeError("tie-aware groups must be TiedGroup")


@dataclass(frozen=True)
class EitherGold:
    """Whole-answer alternatives, in order (a tie between them goes to the first)."""

    alternatives: tuple[tuple[str, tuple[Row, ...]], ...]

    def __post_init__(self) -> None:
        alts = self.alternatives
        items = alts.items() if isinstance(alts, Mapping) else alts  # a mapping, or (name, rows) pairs
        pairs = tuple((name, tuple(tuple(r) for r in rows)) for name, rows in items)
        names = [name for name, _ in pairs]
        if len(pairs) < 2 or len(set(names)) < len(names) or not all(isinstance(n, str) and n for n in names):
            raise ValueError("either-value gold needs at least two alternatives with distinct names")
        object.__setattr__(self, "alternatives", pairs)


@dataclass(frozen=True)
class AnchorGold:
    """A gold (plain rows, a TieAwareGold or an EitherGold) and the anchors' names (name -> IRI, as sorted pairs) the
    question quotes (``anchor_names``)."""

    gold: Any
    anchors: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if isinstance(self.gold, AnchorGold):
            raise ValueError("an anchor-names gold cannot hold another")
        if not isinstance(self.gold, (TieAwareGold, EitherGold)):
            object.__setattr__(self, "gold", tuple(tuple(r) for r in self.gold))
        items = self.anchors.items() if isinstance(self.anchors, Mapping) else self.anchors
        pairs = tuple(sorted((str(name), str(value)) for name, value in items))
        if not pairs or len({name for name, _ in pairs}) < len(pairs):
            raise ValueError("an anchor-names gold needs at least one name, each name once")
        object.__setattr__(self, "anchors", pairs)


Gold = TieAwareGold | EitherGold | AnchorGold


def gold_from_json(obj: Any) -> list[Row] | Gold:
    """A plain table of rows (term JSON or terms), or a ``tie_aware`` / ``either_value`` rule in its JSON form."""
    if isinstance(obj, (TieAwareGold, EitherGold, AnchorGold)):
        return obj
    if not isinstance(obj, Mapping):
        return _rows(obj)
    rule = obj.get("rule")
    if rule == "tie_aware":
        groups = tuple(TiedGroup(g["slots"], _rows(g["members"])) for g in obj["tied_groups"])
        return TieAwareGold(_rows(obj["required"]), groups)
    if rule == "either_value":
        alts = obj["alternatives"]
        if not isinstance(alts, Mapping):
            raise TypeError("either-value alternatives must be an object of named row tables")
        return EitherGold({name: _rows(rows) for name, rows in alts.items()})
    if rule == "anchor_names":
        return AnchorGold(gold_from_json(obj["gold"]), obj["anchors"])
    if rule == "numeric_tolerance":
        raise ValueError("numeric_tolerance scores the gold rows as they are; pass the rows (see question_gold)")
    raise ValueError(f"unknown gold rule {rule!r}")


def _pairs(question: Mapping[str, Any]) -> list[tuple[str, str, bool]]:
    """(name, IRI, homonym) for every anchor the record names: its ``anchors`` (entity, name, homonym) and the pairs of
    its ``params`` by their naming conventions: K with K_name or K_label; entity with name or label; entity_S with
    name_S or label_S; S_entity with S. A pair is a homonym when the record marks its name so (``homonym``, or a
    ``<name key>_homonym`` param)."""
    out: list[tuple[str, str, bool]] = []
    for a in question.get("anchors") or []:
        if isinstance(a, Mapping) and isinstance(a.get("entity"), str) and isinstance(a.get("name"), str):
            out.append((a["name"], a["entity"], a.get("homonym") is True))
    params = question.get("params") or {}
    if not isinstance(params, Mapping):
        return out
    for key, entity in params.items():
        if not isinstance(entity, str) or not entity.startswith(("http://", "https://")):
            continue
        if key == "entity":
            names = ["name", "label"]
        elif key.startswith("entity_"):
            names = [f"name_{key[7:]}", f"label_{key[7:]}"]
        elif key.endswith("_entity"):
            names = [key[:-7]]
        else:
            names = [f"{key}_name", f"{key}_label"]
        for name_key in names:
            if isinstance(params.get(name_key), str):
                out.append((params[name_key], entity, params.get(f"{name_key}_homonym") is True))
    return out


_QUOTED = re.compile(r'"([^"]*)"')


def _quotes(text: str, name: str) -> bool:
    """Whether ``text`` quotes ``name`` as a whole: a complete "..." span of the text, or, for a name that itself holds
    a double quote, the name between quotes."""
    if '"' in name:
        return f'"{name}"' in text
    return name in _QUOTED.findall(text)


def question_anchor_names(question: Mapping[str, Any]) -> dict[str, str]:
    """name -> IRI for the anchors whose name the question text quotes as a whole ("<name>"), where that name belongs
    to one anchor of the record only and the record never marks it a homonym (change-log entry 120)."""
    text = question.get("question") or ""
    by_name: dict[str, set[str]] = {}
    homonyms: set[str] = set()
    for name, entity, homonym in _pairs(question):
        by_name.setdefault(name, set()).add(entity)
        if homonym:
            homonyms.add(name)
    return {name: next(iter(iris)) for name, iris in sorted(by_name.items())
            if len(iris) == 1 and name not in homonyms and name and _quotes(text, name)}


def question_gold(question: Mapping[str, Any]) -> Any:
    """What a question record is scored against, in the JSON form ``run_attempt`` takes as ``gold_rows``.

    Without ``gold_rule`` this is the record's ``gold_rows`` object itself, so such questions score as before. A
    ``tie_aware`` or ``either_value`` rule is put through the scorer's own checks under the question's rules (a
    malformed one fails when the questions are loaded, not at the first scored attempt) and returned as it is. Where
    the question quotes the names of anchors (``question_anchor_names``), either is wrapped in the ``anchor_names``
    form with those names (scorer-rules-v5; with its ``anchor_names`` rule off it scores as the gold alone).
    """
    rule = question.get("gold_rule")
    if rule is None:
        return _with_anchor_names(question, question.get("gold_rows"))
    kind = rule.get("rule") if isinstance(rule, Mapping) else None
    if kind == "numeric_tolerance":
        rules = question.get("scoring_rules") or {}
        if not (rules.get("rel_tol") or rules.get("abs_tol")):
            raise ValueError(f"question {question.get('id')!r}: numeric_tolerance needs a non-zero numeric tolerance")
        return _with_anchor_names(question, question.get("gold_rows"))
    if kind not in ("tie_aware", "either_value"):
        raise ValueError(f"question {question.get('id')!r}: unknown gold rule {kind!r}")
    from .scorer import check_gold  # the scorer imports this module

    rules = question.get("scoring_rules")
    if isinstance(rules, Mapping):
        rules = ScoringRules.from_json(rules) if "version" in rules else ScoringRules(**rules)
    check_gold(gold_from_json(rule), rules if isinstance(rules, ScoringRules) else ScoringRules())
    return _with_anchor_names(question, rule)


def _with_anchor_names(question: Mapping[str, Any], gold: Any) -> Any:
    names = question_anchor_names(question)
    if not names or gold is None:
        return gold
    return {"rule": "anchor_names", "gold": gold, "anchors": names}
