"""Retry feedback in one fixed format, the same for every source (design log, AC2;
contract rule 38), and the matched generic-retry message (AC3).

The feedback lists every ontology-derived check that flagged, deduplicated, in
a fixed order: for each, the KIND of constraint (vocabulary, class membership,
domain or range, disjointness, pattern: rule 38), the violated constraint, and
the offending pattern (static checks) or binding (footprint checks).

Every word shown comes from one fixed table, ``FORMAT_SPEC["text"]``: the
header, the kind labels, one template per (check id, reason) the compiler can
flag, and the fragments that fill a template's placeholders. Only terms of the
query and the data (IRIs, variables, literals, triples) are inserted. A reason
code, a check id or the source's name is never rendered, and the parts of a
check's evidence that describe the source (thresholds, counts, the shared
hierarchy) and the ledger's annotations (data fault, open-world acceptance) are
never read. A flag with no template of its own gets the table's neutral
fallback. Rule 38 bars the words census, observed, count, support, threshold,
data-derived, generated, curated, LLM and model; the framing follows the
prompts' ("A description of this graph's schema follows.").

Number traceability is not fed back: it is ontology-independent (rule 8) and
never triggers a retry. Unknown, not-applicable and passing results are never
fed back either.

The generic control gives the same retry budget with the pre-registered text
"Please re-check your answer." (S0.3, section 5) and no checker information.
Neither message carries an instruction the other lacks (no instruction tail,
change-log entry 30), so the only difference between the arms is the checker's
information.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

from harness.checks.closure import is_datatype
from harness.checks.outcomes import INDEPENDENT_CHECKS, CheckResult, Outcome

FEEDBACK_FORMAT_ID = "checker-feedback-v3"
GENERIC_RETRY_TEXT = "Please re-check your answer."  # S0.3 section 5, byte for byte
# The retry turn's closing sentence in every arm, after the feedback content or the generic text (science, design log,
# change-log entry 126, prereg section 3): a model that read its accepted submission as the attempt's end (the
# submit_report tool says "a valid submission ends the attempt") ended its retry with no submission. Neither the
# feedback format nor the generic text includes it, so their hashes and the tool descriptions are unchanged. Entry 135
# (21 Oct 10:37) reduced it to the minimal fix, final wording: entry 126's sentence also offered to keep the previous
# answer by submitting it unchanged.
SET_ASIDE_TEXT = "Your previous submission has been set aside. Call submit_report again with your final answer."
SET_ASIDE_TEXT_126 = ("Your previous submission has been set aside. Call submit_report again with your final answer; to "
                      "keep your previous answer, submit it unchanged.")
# Every sentence a record's retry turn may close with, so that a record replays with the one its settings name: entry
# 126's (records made before entry 135) and entry 135's.
SET_ASIDE_TEXTS = (SET_ASIDE_TEXT_126, SET_ASIDE_TEXT)

# The fixed order of check types in the feedback (static checks, then footprint checks).
ORDER = (
    "vocabulary.static", "class.static", "domain_range.static", "disjointness.static", "no_common_subclass.static",
    "census_join.static", "census_pattern.static", "class.footprint", "domain_range.footprint",
    "disjointness.footprint", "provenance",
)
# The kind of constraint each check states (rule 38's five kinds).
KIND = {
    "vocabulary.static": "vocabulary",
    "class.static": "class_membership",
    "domain_range.static": "domain_or_range",
    "disjointness.static": "disjointness",
    "no_common_subclass.static": "class_membership",
    "census_join.static": "pattern",
    "census_pattern.static": "pattern",
    "class.footprint": "class_membership",
    "domain_range.footprint": "domain_or_range",
    "disjointness.footprint": "disjointness",
    "provenance": "pattern",
}
FORM = {cid: ("footprint" if cid.endswith(".footprint") or cid == "provenance" else "static") for cid in ORDER}
SAMPLE = 3  # offending values, triples or paths shown per check
# Alternatives or classes listed per list, in sorted IRI order, then ", and N more": the same
# for every source (the study lead's ruling on long lines, design log).
LISTED = 10
# Which part of a requirement a flag's reason is about: a census lists the classes and the
# datatypes allowed at a position as one set of alternatives (rule 20, change-log entry 30).
DATATYPE_REASONS = frozenset({"wrong_datatype", "resource_where_literal_required"})
CLASS_REASONS = frozenset({"outside_required_class", "literal_where_class_required"})


def _resource_templates(cid: str, subject: str, offending: str) -> dict[tuple[str, str], tuple[str, str]]:
    """The five ways a constant or a value can miss a class or datatype requirement; only a
    class miss names the offending constant's classes."""
    bare = offending.split(",")[0]
    return {
        (cid, "outside_required_class"): (f"{subject} must be in {{required}}", offending),
        (cid, "literal_where_class_required"): (f"{subject} must be a resource in {{required}}", bare),
        (cid, "wrong_datatype"): (f"{subject} must have datatype {{required}}", bare),
        (cid, "value_not_in_enumeration"): (f"{subject} must be {{required}}", bare),
        (cid, "resource_where_literal_required"): (f"{subject} must be a literal of {{required}}", bare),
    }


# (check id, reason) -> (violated constraint, offending pattern or binding), with placeholders.
TEMPLATES: dict[tuple[str, str], tuple[str, str]] = {
    ("vocabulary.static", "not_in_source"): (
        "every class and property the query uses must be defined in the schema description", "{term}"),
    **_resource_templates("class.static", "the constant", "{constant}, whose classes are {types}"),
    **_resource_templates("domain_range.static", "{position}", "{constant}, whose classes are {types}"),
    ("domain_range.static", "literal_in_resource_position"): (
        "a value that a range requires to be a literal cannot also be used as a resource", "{variable}"),
    ("disjointness.static", "explicit_disjointness"): (
        "the classes required for {variable} are disjoint", "{classes}"),
    ("no_common_subclass.static", "no_common_named_subclass"): (
        "no class is a subclass of every class required for {variable}", "{classes}"),
    ("census_join.static", "census_low_support"): (
        "the schema description lists no class allowed in all of these positions of {variable} together",
        "{positions}"),
    ("census_pattern.static", "census_empty_pattern"): (
        "the schema description lists no such pattern for this property", "{pattern}"),
    **_resource_templates("class.footprint", "every value of {variable}", "{values}"),
    **_resource_templates("domain_range.footprint", "every value in {position}", "{values}"),
    ("disjointness.footprint", "disjoint_asserted_types"): (
        "no value of {variable} may be a member of two disjoint classes", "{witnessed}"),
    ("provenance", "no_provenance_path"): (
        ("every answer value of a class that requires a path must reach a member of the required class through the "
         "required property"), "{paths}"),
}
FALLBACK = ("the query does not meet a constraint of the schema description", "{key}")

TEXT = {
    "header": "Your answer was checked against a description of this graph's schema. {n} {checks} failed:",
    "checks": {"one": "check", "many": "checks"},
    "item": "{i}. {line}",
    "line": "{label} ({where}): {constraint}{branch}. Offending {offending_kind}: {offending}.",
    "labels": {"vocabulary": "Vocabulary", "class_membership": "Class membership",
               "domain_or_range": "Domain or range", "disjointness": "Disjointness", "pattern": "Pattern"},
    "where": {"static": "query pattern", "footprint": "query results"},
    "offending_kind": {"static": "pattern", "footprint": "binding"},
    "templates": {f"{cid} {reason}": list(t) for (cid, reason), t in TEMPLATES.items()},
    "fallback": list(FALLBACK),
    "branch": {"optional": " (inside an OPTIONAL part)", "negated": " (inside a negated part)",
               "across optional groups": " (across OPTIONAL parts)", "other": " (inside an OPTIONAL or negated part)"},
    "position": {"subject-of": "the subject of {property}", "object-of": "the object of {property}"},
    "alternatives": " or ",
    "conjuncts": " and ",
    "enumeration": "one of {values}",
    "required_unknown": "the required class",
    "no_class": "{constant}, which has no class",
    "more": "{shown}, and {more} more",
    "values": {"one": "{flagged} of {total} value, for example {sample}",
               "many": "{flagged} of {total} values, for example {sample}"},
    "values_without_sample": {"one": "{flagged} of {total} value", "many": "{flagged} of {total} values"},
    "origins": {"type pattern": "{class} (type pattern)", "domain of": "{class} (domain of {property})",
                "range of": "{class} (range of {property})", "other": "{class}"},
    "positions": {"class": "member of {class}", "subject": "subject of {property}", "object": "object of {property}"},
    "path": "{value} through {property} to {class}",
    "list_separator": "; ",
    "item_separator": ", ",
    "generic": GENERIC_RETRY_TEXT,
}

FORMAT_SPEC = {"id": FEEDBACK_FORMAT_ID, "order": ORDER, "kind": KIND, "form": FORM, "sample": SAMPLE,
               "listed": LISTED, "datatype_reasons": sorted(DATATYPE_REASONS),
               "class_reasons": sorted(CLASS_REASONS), "text": TEXT}


def format_sha256() -> str:
    """The hash of everything that fixes the format (canonical JSON of FORMAT_SPEC: sorted
    keys, no spaces, UTF-8); recorded with every checker decision."""
    text = json.dumps(FORMAT_SPEC, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generic_sha256() -> str:
    return hashlib.sha256(GENERIC_RETRY_TEXT.encode("utf-8")).hexdigest()


def set_aside_sha256(text: str | None = None) -> str:
    """The sha256 of a set-aside sentence: ``text``, else this code's (SET_ASIDE_TEXT)."""
    return hashlib.sha256((SET_ASIDE_TEXT if text is None else text).encode("utf-8")).hexdigest()


def set_aside_text(sha256: Any) -> str | None:
    """The known set-aside sentence (SET_ASIDE_TEXTS) whose sha256 is ``sha256``, or None."""
    return next((t for t in SET_ASIDE_TEXTS if set_aside_sha256(t) == sha256), None)


def retry_turn_text(content: str, set_aside: bool | str) -> str:
    """The retry turn's text: the feedback content or the generic text, then (entry 126) the set-aside sentence: this
    code's when ``set_aside`` is True, the one given when it is a sentence (a record's, replayed), none when False."""
    if not set_aside:
        return content
    return f"{content}\n\n{SET_ASIDE_TEXT if set_aside is True else set_aside}"


# -- the OBQC arm's lines (harness/baselines/obqc.py, design log) ----------------------------
#
# The OBQC arm's flags are its eight published rules, which have no template in the table
# above. Their lines use the paper's own explanation templates (arXiv:2405.11706v1, sections
# 3.1-3.2) in this format's line structure, with two changes so they meet rule 38 and
# change-log entry 30 like every other line: the instruction tails are cut ("Please only use
# properties from the ontology, ...", "your output should be something human readable, an ID or
# a label") and "the ontology" reads "the schema description", the prompts' framing. Kept in a
# table of its own, with its own id and hash, so the source arms' format (and every record
# made with it) is unchanged. Approved by the study lead (design log).

OBQC_FORMAT_ID = "obqc-feedback-v1"
OBQC_ORDER = ("obqc.domain", "obqc.range", "obqc.double_range", "obqc.double_domain", "obqc.domain_range",
              "obqc.incorrect_property", "obqc.iri_output", "obqc.subject_output")
OBQC_KIND = {"obqc.domain": "domain_or_range", "obqc.range": "domain_or_range",
             "obqc.double_range": "domain_or_range", "obqc.double_domain": "domain_or_range",
             "obqc.domain_range": "domain_or_range", "obqc.incorrect_property": "vocabulary",
             "obqc.iri_output": "domain_or_range", "obqc.subject_output": "pattern"}
OBQC_TEMPLATES = {
    "obqc.domain": "The property {p} has domain {domain}, but its subject {s} is a {class}, which isn't a subclass of "
                   "{domain}",
    "obqc.range": "The property {p} has range {range}, but its object {o} is a {class}, which isn't a subclass of "
                  "{range}",
    "obqc.double_range": "The property {p} has range {rangep}, and {q} has range {rangeq}, and these are incompatible",
    "obqc.double_domain": "The property {p} has domain {domp}, and {q} has domain {domq}, and these are incompatible",
    "obqc.domain_range": ("The property {p} has range {rangep}, and {q} has domain {domq}, and these are "
                          "incompatible with the query"),
    "obqc.incorrect_property": "The property {p} isn't defined in the schema description",
    "obqc.iri_output": "Your selected variable {varname} is an IRI",
    "obqc.subject_output": "Your selected variable ?{varname} is an IRI (the subject of a triple is always an IRI)",
}
OBQC_TEXT = {"header": TEXT["header"], "checks": TEXT["checks"], "item": TEXT["item"],
             "line": "{label} ({where}): {sentence}.", "labels": TEXT["labels"], "where": TEXT["where"]["static"]}
OBQC_FORMAT_SPEC = {"id": OBQC_FORMAT_ID, "order": OBQC_ORDER, "kind": OBQC_KIND, "templates": OBQC_TEMPLATES,
                    "text": OBQC_TEXT}


def obqc_format_sha256() -> str:
    text = json.dumps(OBQC_FORMAT_SPEC, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def order_index(check_id: str) -> int:
    """A check's place in the fixed order: the source checks, then the OBQC rules."""
    return ORDER.index(check_id) if check_id in ORDER else len(ORDER) + OBQC_ORDER.index(check_id)


def _obqc_line(r: CheckResult) -> str:
    """The rule's template filled from the flagged binding (terms rendered by obqc.py: full
    IRIs in angle brackets, query variables as ?name, literals as written)."""
    sentence = OBQC_TEMPLATES[r.check_id].format(**r.evidence["binding"])
    return OBQC_TEXT["line"].format(label=OBQC_TEXT["labels"][OBQC_KIND[r.check_id]], where=OBQC_TEXT["where"],
                                    sentence=sentence)


# -- rendering terms ---------------------------------------------------------------------


def _term(text: str) -> str:
    """A full IRI in angle brackets; a variable, literal or blank node as written."""
    t = text.strip()
    if not t or t[0] in "?\"'_<" or t.startswith("urn:www2027"):
        return t
    return f"<{t}>"


def _listed(items: Iterable, joiner: str, render=_term) -> str:
    """At most LISTED items, in sorted order, then how many more."""
    ordered = sorted(items)
    shown = joiner.join(render(x) for x in ordered[:LISTED])
    if len(ordered) > LISTED:
        shown = TEXT["more"].format(shown=shown, more=len(ordered) - LISTED)
    return shown


def _alternatives(text: str, keep=None) -> str:
    """'A|B|...' as alternatives; ``keep`` narrows them to the part a flag is about, unless
    that would leave none."""
    members = text.split("|")
    if keep is not None:
        members = [m for m in members if keep(m)] or members
    return _listed(members, TEXT["alternatives"])


def _required(items: Iterable[str], reason: str | None = None) -> str:
    keep = is_datatype if reason in DATATYPE_REASONS else (
        (lambda m: not is_datatype(m)) if reason in CLASS_REASONS else None)
    parts = []
    for item in items:
        if item.startswith("<unsupported"):
            continue
        if item.startswith("oneOf(") and item.endswith(")"):
            values = _listed(item[6:-1].split(","), TEXT["item_separator"], render=str)
            parts.append(TEXT["enumeration"].format(values=values))
        else:
            parts.append(_alternatives(item, keep))
    return TEXT["conjuncts"].join(parts) if parts else TEXT["required_unknown"]


def _position(pos: str) -> str:
    """'subject-of P' / 'object-of P' -> words."""
    kind, _, what = pos.partition(" ")
    template = TEXT["position"].get(kind)
    return template.format(property=_alternatives(what)) if template else pos


def _branch(r: CheckResult) -> str:
    if not r.branch_local:
        return ""
    scope = r.evidence.get("scope") if r.evidence.get("scope") in TEXT["branch"] else r.scope
    return TEXT["branch"].get(scope, TEXT["branch"]["other"])


def _class_with_origin(pair: tuple[str, str]) -> str:
    expr, origin = pair
    cls = _alternatives(expr)
    if origin == "type pattern":
        return TEXT["origins"]["type pattern"].format(**{"class": cls})
    for side in ("domain of", "range of"):
        if origin.startswith(side + " "):
            return TEXT["origins"][side].format(**{"class": cls, "property": _alternatives(origin[len(side) + 1:])})
    return TEXT["origins"]["other"].format(**{"class": cls})


def _classes(r: CheckResult) -> str:
    implied, origins = r.evidence.get("implied") or [], r.evidence.get("origins") or []
    pairs = list(zip(implied, list(origins) + [""] * len(implied)))
    return _listed(pairs, TEXT["list_separator"], render=_class_with_origin)


def _values(r: CheckResult) -> str:
    nodes = r.evidence.get("nodes") or {}
    flagged = [n for n, v in nodes.items() if v == "flag"][:SAMPLE]
    counts = r.evidence.get("node_counts") or {}
    total = sum(counts.values()) if counts else len(nodes)
    number = "one" if total == 1 else "many"
    fields = {"flagged": counts.get("flag", len(flagged)), "total": total,
              "sample": TEXT["item_separator"].join(_term(n) for n in flagged)}
    return (TEXT["values"] if flagged else TEXT["values_without_sample"])[number].format(**fields)


def _types(r: CheckResult) -> str:
    return _listed(r.evidence.get("asserted_types") or [], TEXT["item_separator"])


def _census_position(p: str) -> str:
    head, _, tail = p.partition(" ")
    if p.endswith(" class"):
        return TEXT["positions"]["class"].format(**{"class": _alternatives(p[: -len(" class")])})
    return TEXT["positions"][head].format(property=_alternatives(tail))


def _census_positions(r: CheckResult) -> str:
    positions = [p for p in r.evidence.get("positions") or []
                 if p.endswith(" class") or p.partition(" ")[0] in ("subject", "object")]
    return _listed(positions, TEXT["list_separator"], render=_census_position)


def _paths(r: CheckResult) -> str:
    shown = []
    for x in r.evidence.get("requirements") or []:
        text, _, outcome = x.rpartition(": ")
        parts = text.split(" ")
        if outcome == "flag" and len(parts) == 4 and parts[2] == "some":
            shown.append(TEXT["path"].format(**{"value": _term(parts[0]), "property": _term(parts[1]),
                                                "class": _term(parts[3])}))
    return TEXT["list_separator"].join(shown[:SAMPLE])


def _fields(r: CheckResult) -> dict[str, str]:
    """The placeholder values of one flag: terms of the query and the data only."""
    cid, key = r.check_id, r.key
    fields = {"key": _term(key), "required": _required(r.evidence.get("required") or [], r.reason), "variable": key}
    if cid in ("class.static", "domain_range.static") and r.reason != "literal_in_resource_position":
        parts = key.rsplit(" ", 2)
        const, pos = (parts[0], f"{parts[1]} {parts[2]}") if len(parts) == 3 else (key, "")
        fields.update(constant=_term(const), position=_position(pos), types=_types(r))
    elif cid == "class.footprint":
        var, _, cls = key.partition(" a ")
        fields.update(variable=var, required=_required(r.evidence.get("required") or [cls], r.reason),
                      values=_values(r))
    elif cid == "domain_range.footprint":
        fields.update(position=_position(key), values=_values(r))
    elif cid == "vocabulary.static":
        fields.update(term=_term(key))
    elif cid == "census_pattern.static":
        subject, pred, cls = (key.split(" ", 2) + ["", ""])[:3]
        fields.update(pattern=" ".join((_term(subject), _term(pred), _term(cls))))
    elif cid == "provenance":
        fields.update(paths=_paths(r))
    fields.update(classes=_classes(r), positions=_census_positions(r),
                  witnessed=TEXT["list_separator"].join((r.evidence.get("witnessed") or [])[:SAMPLE]))
    return fields


def _line(r: CheckResult) -> str:
    if r.check_id in OBQC_KIND:
        return _obqc_line(r)
    fields = _fields(r)
    constraint, offending = TEMPLATES.get((r.check_id, r.reason), FALLBACK)
    if "{types}" in offending and not fields.get("types"):
        offending = TEXT["no_class"]
    form = FORM[r.check_id]
    return TEXT["line"].format(label=TEXT["labels"][KIND[r.check_id]], where=TEXT["where"][form],
                               constraint=constraint.format(**fields), branch=_branch(r),
                               offending_kind=TEXT["offending_kind"][form], offending=offending.format(**fields))


def fed_back(results: Iterable[CheckResult]) -> list[CheckResult]:
    """The results that enter the feedback: ontology-derived flags only."""
    return [r for r in results if r.outcome is Outcome.FLAG and r.check_id not in INDEPENDENT_CHECKS]


def fired_lines(results: Iterable[CheckResult]) -> list[str]:
    """One line per distinct fired check, in the fixed order (check type, then text)."""
    keyed = {(order_index(r.check_id), line): line for r in fed_back(results) for line in [_line(r)]}
    return [keyed[k] for k in sorted(keyed)]


def render_feedback(results: Iterable[CheckResult]) -> str:
    return render_lines(fired_lines(results))


def render_lines(lines: list[str]) -> str:
    """The feedback message from the fired lines (``fired_lines``, or a verdict's ``fired``)."""
    if not lines:
        raise ValueError("no ontology-derived check flagged: there is nothing to feed back")
    n = len(lines)
    head = TEXT["header"].format(n=n, checks=TEXT["checks"]["one" if n == 1 else "many"])
    body = "\n".join(TEXT["item"].format(i=i, line=line) for i, line in enumerate(lines, 1))
    return f"{head}\n{body}"
