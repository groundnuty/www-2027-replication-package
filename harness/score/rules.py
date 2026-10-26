"""The scoring rules, frozen before any run and recorded in every run record.

Everything that can change a score is a field here, and ``to_json`` serialises
all of it together with ``RULES_VERSION``. A change to the *semantics* of the
rules (as opposed to a different setting of a field) must bump ``RULES_VERSION``.

scorer-rules-v5 (science ruling on issue 3, 21 Oct 02:48, change-log entry 120) adds three rules, each a field and on
by default: ``extra_columns``, ``anchor_names`` and ``temporal_strings`` (scorer.py says what each does). With all three
off the rules are exactly scorer-rules-v4 and serialise as the v4 JSON, so a record scored under v4 reads back as v4
and writes the same, and anything frozen under v4 can still be scored as it was.

scorer-rules-v6 (change-log entry 127; a pre-freeze cross-review on issue 3, 07:46) narrows ``extra_columns``: a choice
of the answer's columns is used only where exactly one order-kept choice matches the gold's column kinds (the field
``extra_columns_by_kind``, on by default). A v5 rules object, as stored in a v5 score, reads back with it off and
writes the same v5 JSON, so a v5 score is reproduced by its own rules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Literal

RULES_VERSION = "scorer-rules-v6"
RULES_VERSION_V5 = "scorer-rules-v5"
RULES_VERSION_V4 = "scorer-rules-v4"
V5_FIELDS = ("extra_columns", "anchor_names", "temporal_strings")
V6_FIELDS = ("extra_columns_by_kind",)


@dataclass(frozen=True)
class ScoringRules:
    """Settings of the answer scorer.

    multiset      False: set semantics, rows that are equal under the canonical
                  form collapse to one on both sides before matching.
                  True: duplicate rows count, a row returned twice is two rows.
    rel_tol       Relative tolerance of numeric comparison (``math.isclose``).
    abs_tol       Absolute tolerance of numeric comparison (``math.isclose``);
                  it is what lets a value near zero match zero.
    column_order  "any": when the answer has the same arity as the gold rows
                  (and at most four columns) the permutation of the answer's
                  columns that matches the most rows is used.
                  "fixed": the answer's columns must come in the gold order.
    extra_columns     (v5) an answer whose rows all have more columns than the gold is also scored on a choice of as
                      many of its columns as the gold has (which choice: ``extra_columns_by_kind``).
    anchor_names      (v5) against a gold of the "anchor_names" form (gold.py), an untagged answer literal equal to the
                      name by which the question names one of its anchors is read as that anchor's IRI, in a column
                      whose bound gold cells are all IRIs.
    temporal_strings  (v5) an untagged answer literal that is a valid xsd:date (xsd:dateTime) is read with that
                      datatype in a column whose bound gold cells all have it.
    extra_columns_by_kind  (v6) True: the one order-kept choice whose column kinds (IRI, numeric, literal) match the
                      gold's, and none where no choice or more than one does; False (v5): the best-scoring choice of
                      every order-kept choice.
    """

    multiset: bool = False
    rel_tol: float = 1e-6
    abs_tol: float = 1e-9
    column_order: Literal["any", "fixed"] = "any"
    extra_columns: bool = True
    anchor_names: bool = True
    temporal_strings: bool = True
    extra_columns_by_kind: bool = True

    def __post_init__(self) -> None:
        for name in ("rel_tol", "abs_tol"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a non-negative number, got {value!r}")
        if self.column_order not in ("any", "fixed"):
            raise ValueError(f"column_order must be 'any' or 'fixed', got {self.column_order!r}")
        for name in ("multiset", *V5_FIELDS, *V6_FIELDS):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool, got {getattr(self, name)!r}")
        if self.is_v4 and not self.extra_columns_by_kind:  # moot with every v5 rule off: one v4 object, not two
            object.__setattr__(self, "extra_columns_by_kind", True)

    @property
    def is_v4(self) -> bool:
        """All three v5 rules off: exactly scorer-rules-v4."""
        return not any(getattr(self, name) for name in V5_FIELDS)

    def as_v4(self) -> ScoringRules:
        """These settings with every v5 rule off: the scorer-rules-v4 semantics, which the frozen instruments keep
        (the fault banks' labels, the checker's rule 22; change-log entry 120)."""
        return replace(self, **dict.fromkeys(V5_FIELDS, False))

    @property
    def version(self) -> str:
        if self.is_v4:
            return RULES_VERSION_V4
        return RULES_VERSION if self.extra_columns_by_kind else RULES_VERSION_V5

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "version": self.version,
            "multiset": self.multiset,
            "rel_tol": self.rel_tol,
            "abs_tol": self.abs_tol,
            "column_order": self.column_order,
        }
        if not self.is_v4:
            out.update({name: getattr(self, name) for name in V5_FIELDS})
        if out["version"] == RULES_VERSION:
            out.update({name: getattr(self, name) for name in V6_FIELDS})
        return out

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> ScoringRules:
        version = obj.get("version")
        base = (obj["multiset"], obj["rel_tol"], obj["abs_tol"], obj["column_order"]) if version in (
            RULES_VERSION_V4, RULES_VERSION_V5, RULES_VERSION) else None
        if version == RULES_VERSION_V4:
            if any(name in obj for name in (*V5_FIELDS, *V6_FIELDS)):
                raise ValueError(f"a {RULES_VERSION_V4} rules object has no {', '.join((*V5_FIELDS, *V6_FIELDS))}")
            rules = cls(*base, False, False, False)
        elif version == RULES_VERSION_V5:
            if any(name in obj for name in V6_FIELDS):
                raise ValueError(f"a {RULES_VERSION_V5} rules object has no {', '.join(V6_FIELDS)}")
            rules = cls(*base, *(obj[name] for name in V5_FIELDS), False)
        elif version == RULES_VERSION:
            rules = cls(*base, *(obj[name] for name in V5_FIELDS), *(obj[name] for name in V6_FIELDS))
        else:
            raise ValueError(f"rules version {version!r} is not {RULES_VERSION!r}, {RULES_VERSION_V5!r} or "
                             f"{RULES_VERSION_V4!r}")
        if rules.version != version:
            # to_json writes these rules under another version (every v5 rule off is v4; a v6 object with the kind
            # rule off is v5), so this object is not one it wrote
            raise ValueError(f"these {version} rules are written as {rules.version}")
        if rules.to_json() != obj:  # a field to_json does not write would be dropped: the object could not round-trip
            raise ValueError(f"a {version} rules object has only {', '.join(rules.to_json())}, got "
                             f"{', '.join(map(str, obj))}")
        return rules
