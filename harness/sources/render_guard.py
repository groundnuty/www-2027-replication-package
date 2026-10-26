"""No prompt from a render whose budget no ruling gives (external reviewer 6 review of a929c39, item 3).

The render budget per graph is a design ruling (change-log entry 88, design log, 20 Oct 18:16: 27,000 o200k tokens on
Rhea and opaque Rhea, fit-bound; design log, entries 22, 56 and 60 and 19 Oct 15:49: 50,000 on UniProt), recorded as
``StoreLayout.render_budget``.
``build_sources`` records in each render manifest whether the budget it rendered at was the graph's ruled one
(``budget_ruled``); prompt assembly reads a built render only through ``load_ruled_render``, and a builder that
renders a graph itself asks ``ruled_budget`` first. A manifest from before the flag counts as unruled.

Kept apart from ``harness.sources.render``, which stays byte-identical with feat/prompt-conditions.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from harness.sources.vocab import RHEA_FIT, StoreLayout

STEP = 1000
# The rules a ruled budget may come from (External review of b585c0a and 3831e43, item 2: a configured number is not
# enough, the guard recomputes the rule's value on the current sources and refuses any other number):
RULES = {
    "entry-22": "change-log entry 22: the largest own content among the curated source and the generated draws 1 to 3, "
                "rounded up to the next 1,000",
    "no-truncation": "design log, 19 Oct 15:49: the largest own content of any prompted source, the census included, "
                     "rounded up to the next 1,000, so that no source is truncated",
    "fit-bound": "change-log entry 88: the largest budget at which every model fits entry 60's rule, as the committed "
                 "fit computation the layout names gives it (scripts/fit_budget.py), measured on the sources being "
                 "rendered, never above the no-truncation value",
}
_PROMPTED = re.compile(r"curated|aligned-curated|census-t5|generated-[123]")
FIT_SCHEMA = "fit-budget/1"  # scripts/fit_budget.py compute's output


@dataclass(frozen=True)
class Ruling:
    """A graph's ruled budget as science ruled it: the rule, the number, where it was ruled and, for a fit-bound
    number, the committed fit computation that gives it (path from the repository root, sha256), the graphs that one
    budget covers and the reading of entry 60 it applies."""

    rule: str
    budget: int
    citation: str
    fit: tuple[str, str] | None = None
    fit_graphs: tuple[str, ...] = ()
    fit_reading: str = "strict"
    # The ruling lets a source's own content be cut at this budget (a fit-bound number below entry 22's value, the
    # truncation reported per arm); without it a build refuses a render that loses own content.
    truncates_own: bool = False
    # A fit-bound ruling's reservation and per-model tool caps, as ruled: the fit computation must have used exactly
    # these (None: not pinned beyond what the computation records).
    fit_reservation: int | None = None
    fit_tool_caps: tuple[tuple[str, int], ...] | None = None


# Every ruled budget, keyed by the layout's name (External review of 8a224a3, items 2 and 3: a number is authorised by
# its ruling, never by a recomputation alone). The layout must name the same number, rule and fit computation.
RULINGS: dict[str, Ruling] = {
    # Change-log entry 88 (design log, 20 Oct 18:16): one budget for Rhea and opaque Rhea, the largest at which every model
    # in P fits entry 60's rule by the strict reading, as the committed fit computation gives it (27,000; Gemma binding
    # on the cap-audit proxy loop at its 62,000 cap); the census is cut by support, curated and the draws are whole.
    "rhea-142": Ruling("fit-bound", 27000, "change-log entry 88 (design log, 20 Oct 18:16)", RHEA_FIT,
                       ("rhea-142", "opaque-rhea-142")),
    "opaque-graph-1": Ruling("fit-bound", 27000, "change-log entry 88 (design log, 20 Oct 18:16)", RHEA_FIT,
                             ("rhea-142", "opaque-rhea-142")),
    "uniprot-sp-2026_03": Ruling("no-truncation", 50000,
                                 "design log, change-log entries 22, 56 and 60, and the ruling of 19 Oct 15:49"),
    # Change-log entry 94 (design log, 20 Oct 00:01): no truncation, as entries 84 and 88 rule elsewhere: the largest own
    # content of any prompted source, the census's 3,343 o200k, rounded up (entry 22's value would be 3,000, from draw
    # 3's 2,665, and would cut the census); every model in P fits at 0.95W.
    "text2sparql-ck26": Ruling("no-truncation", 4000, "change-log entry 94 (design log, 20 Oct 00:01)"),
    # a15 (design log, 20 Oct): with imported files rendered after the source's own layers (change-log entry 101), the
    # largest own content among both curated arms and the drafts, rounded up, is below the largest budget every model
    # in P fits at the ruled caps (design log, 19 Oct 22:34; design log, 20 Oct 12:54 and 13:23), so B is that value and
    # nothing is cut, as for the corporate graph under entry 94.
    "a15": Ruling("no-truncation", 29000, "design log, 20 Oct (rule 16 as amended, change-log entry 101; design log, "
                  "19 Oct 22:34)"),
}


class UnruledRender(ValueError):
    """A render, or a graph, whose budget no ruling gives."""


def ruled_budget(layout: StoreLayout) -> int:
    """The graph's ruled render budget; refuses a graph without one."""
    if layout.render_budget is None:
        raise UnruledRender(f"{layout.name}: no ruled render budget (StoreLayout.render_budget is unset); no prompt "
                            "is built from a render of this graph until science rules one")
    return layout.render_budget


def round_up(tokens: int) -> int:
    return -(-tokens // STEP) * STEP


def budget_values(own: dict[str, int]) -> dict[str, int]:
    """Entry 22's value and the no-truncation value from the own content (o200k tokens) of a graph's sources by name;
    only the prompted ones count (curated, census-t5, generated-1 to 3). Entry 22's value needs the curated source or
    a draw; without one it is left out."""
    own = {k: v for k, v in own.items() if _PROMPTED.fullmatch(k)}
    measured = [v for k, v in own.items() if k != "census-t5"]
    out = {"entry-22": round_up(max(measured))} if measured else {}
    if own:
        out["no-truncation"] = round_up(max(own.values()))
    return out


def _committed(path: Path, root: Path | None = None) -> str | None:
    """None when ``path`` is tracked and unmodified at HEAD of the git repository holding it, or, in a deployed copy
    without git (``root``, the copy's code root), when the copy's TREE binds its COMMIT, its tree digest recomputes
    from the files on disk (harness.agent.replay_check, the one implementation the runners verify) and that digest
    covers ``path``; else why not."""
    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(path.parent), *args], capture_output=True, check=False)

    if root is not None and not _git_checkout(Path(root)):  # a deployed copy: only its own verified tree counts
        return _deployed(path, root)
    top = git("rev-parse", "--show-toplevel")
    if top.returncode != 0:
        return "it is in no git repository"
    rel = path.resolve().relative_to(Path(top.stdout.decode().strip()).resolve()).as_posix()
    head = git("cat-file", "-p", f"HEAD:{rel}")
    if head.returncode != 0:
        return "it is not committed at HEAD"
    if head.stdout != path.read_bytes():
        return "it differs from its committed version at HEAD"
    return None


def _git_checkout(root: Path) -> bool:
    return subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"], capture_output=True,
                          check=False).returncode == 0


def _deployed(path: Path, root: Path) -> str | None:
    """Why ``path`` is not part of the verified deployed copy at ``root`` (no git): None when it is."""
    from harness.agent import replay_check

    root = Path(root)
    if not (root / "TREE").is_file() or not (root / "COMMIT").is_file():
        return "it is in no git repository, and its deployed copy has no COMMIT and TREE (scripts/write_tree_digest.py)"
    code = replay_check.code_commit(root)
    if code.get("source") != "COMMIT" or code.get("git_dirty") is not False:
        return ("the deployed copy's TREE does not bind its COMMIT, or its files changed after TREE was written; "
                "deploy it again")
    try:
        rel = Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return "it is outside the deployed copy"
    if rel not in replay_check.code_files(root):
        return "the deployed copy's tree digest has not covered it (not among the code paths TREE digests)"
    return None


def authorised_budget(layout: StoreLayout, values: dict[str, int], root: Path,
                      sources: dict[str, dict[str, str]] | None = None,
                      owner_namespaces_sha256: dict[str, str | None] | None = None) -> int:
    """The graph's ruled budget, refused unless the layout names exactly the ruling recorded for it (RULINGS) and the
    current sources (``values``: ``budget_values`` of their own content) still bear it out: entry 22's value equals
    the number; a no-truncation number still renders every source whole (the recomputed value is at most the number);
    a fit-bound number is what its committed fit computation (resolved against ``root``) gives, never above the
    no-truncation value, and that computation measured the sources being rendered (``sources``: the graph, as the
    fit's sizes name it, to its sources' ``source_file.content_digests``)."""
    budget = ruled_budget(layout)
    ruling = RULINGS.get(layout.name)
    if ruling is None:
        raise UnruledRender(f"{layout.name}: no ruling is recorded for the budget {budget} (render_guard.RULINGS)")
    rule = ruling.rule
    if (budget, layout.render_budget_rule, layout.render_budget_fit) != (ruling.budget, rule, ruling.fit):
        raise UnruledRender(f"{layout.name}: the layout's budget {budget} ({layout.render_budget_rule}) is not the "
                            f"ruled {ruling.budget} ({rule}, {ruling.citation})")
    if rule == "entry-22":
        if values.get("entry-22") != budget:
            raise UnruledRender(f"{layout.name}: the ruled budget {budget} is not entry 22's value on the current "
                                f"sources ({values.get('entry-22')}; measured {values}); the sources changed since "
                                "the ruling")
        return budget
    if "no-truncation" not in values:
        raise UnruledRender(f"{layout.name}: the no-truncation value cannot be recomputed: its sources were not "
                            "measured")
    if rule == "no-truncation":
        if values["no-truncation"] > budget:
            raise UnruledRender(f"{layout.name}: the ruled budget {budget} (no-truncation) no longer renders every "
                                f"source whole (the no-truncation value is {values['no-truncation']}; measured "
                                f"{values})")
        return budget
    if budget > values["no-truncation"]:
        raise UnruledRender(f"{layout.name}: the fit-bound budget {budget} is above the no-truncation value "
                            f"({values['no-truncation']}), which already renders every source whole")
    if ruling.fit is None:
        raise UnruledRender(f"{layout.name}: a fit-bound budget {budget} without the fit computation that gives it")
    rel, sha = ruling.fit
    fit = _committed_json(layout, Path(root), rel, sha, "the fit computation")
    if (fit.get("schema") != FIT_SCHEMA or sorted(fit.get("graphs") or []) != sorted(ruling.fit_graphs)
            or fit.get("reading") != ruling.fit_reading):
        raise UnruledRender(f"{layout.name}: {rel} is not a scripts/fit_budget.py computation ({FIT_SCHEMA}) over "
                            f"the graphs {sorted(ruling.fit_graphs)} by the {ruling.fit_reading} reading")
    files = fit.get("input_files") or {}
    if set(files) != {"sizes", "models"}:
        raise UnruledRender(f"{layout.name}: {rel} records no inputs to recompute it from (input_files: the sizes and "
                            "the models, each with its sha256)")
    inputs = {key: _committed_json(layout, Path(root), files[key].get("path"), files[key].get("sha256"),
                                   f"the {key} file of {rel}") for key in ("sizes", "models")}
    from harness.sources import fit as fit_computation  # the computation fit_budget.py ran (External review of ee6fd40)

    try:
        again = fit_computation.compute(inputs["sizes"], inputs["models"], fit.get("reservation"),
                                        fit.get("reading"), fit.get("tool_caps"))
    except (fit_computation.FitError, KeyError, TypeError) as e:
        raise UnruledRender(f"{layout.name}: the inputs {rel} records do not recompute: {e}") from None
    differ = [k for k in ("budget", "reservation", "reading", "tool_caps", "graphs") if again.get(k) != fit.get(k)]
    if differ:
        raise UnruledRender(f"{layout.name}: {rel} does not recompute from its inputs ({differ}: recorded "
                            f"{[fit.get(k) for k in differ]}, recomputed {[again.get(k) for k in differ]})")
    if fit.get("budget") != budget:
        raise UnruledRender(f"{layout.name}: the fit computation {rel} gives {fit.get('budget')}, not the ruled "
                            f"{budget}")
    if ruling.fit_reservation is not None and fit.get("reservation") != ruling.fit_reservation:
        raise UnruledRender(f"{layout.name}: {rel} reserves {fit.get('reservation')} output tokens, not the ruled "
                            f"{ruling.fit_reservation}")
    if ruling.fit_tool_caps is not None and fit.get("tool_caps") != dict(ruling.fit_tool_caps):
        raise UnruledRender(f"{layout.name}: {rel} was computed at tool caps {fit.get('tool_caps')}, not the ruled "
                            f"{dict(ruling.fit_tool_caps)}")
    _measured_as_rendered(layout, rel, inputs["sizes"], ruling.fit_graphs, sources, owner_namespaces_sha256)
    return budget


def _measured_as_rendered(layout: StoreLayout, rel: str, sizes: dict, fit_graphs: tuple[str, ...],
                          sources: dict[str, dict[str, str]] | None,
                          owner_namespaces_sha256: dict[str, str | None] | None = None) -> None:
    """Refuses a fit whose sizes were measured on other sources than the ones being rendered (External review of bebd077,
    item 2): the same budget values can hide a source whose prompt counts more model tokens. Each measured arm's
    source (its name and content sha256) must be the source of that name being rendered now."""
    if not sources:
        raise UnruledRender(f"{layout.name}: a fit-bound budget needs the sources being rendered, to compare with "
                            f"those {rel} was measured on")
    for graph, current in sources.items():
        if graph not in fit_graphs:
            raise UnruledRender(f"{layout.name}: {rel} does not cover {graph} (it covers {sorted(fit_graphs)})")
        arms = (sizes.get("graphs") or {}).get(graph) or {}
        recorded = (sizes.get("inputs") or {}).get(graph) or {}
        if not arms or set(recorded) != set(arms) or not all(r.get("source") and r.get("content_sha256")
                                                             for r in recorded.values()):
            raise UnruledRender(f"{layout.name}: the sizes of {rel} do not record the source each {graph} arm was "
                                "measured on (its name and content_sha256); remeasure them (scripts/fit_budget.py "
                                "measure) and recompute the fit")
        stale = [f"{r['source']} ({arm}) measured as {r['content_sha256'][:12]}..., rendered as "
                 f"{(current.get(r['source']) or 'none')[:12]}" for arm, r in sorted(recorded.items())
                 if current.get(r["source"]) != r["content_sha256"]]
        if stale:
            raise UnruledRender(f"{layout.name}: the {graph} sources being rendered are not the ones {rel} was "
                                f"measured on: {'; '.join(stale)}; remeasure the sizes and recompute the fit")
        # A layout with extra arms (a15) has its fit cover every rendered source, the aligned arm and drafts 2 and 3
        # included (External reviews of 98ebdc1); elsewhere the measured arms bound the fit as before.
        unmeasured = sorted(set(current) - {r["source"] for r in recorded.values()}) if layout.extra_arms else []
        measured_ns = ((sizes.get("imported") or {}).get(graph) or {}).get("owner_namespaces_sha256")
        if measured_ns != (owner_namespaces_sha256 or {}).get(graph):  # rule 16 as amended orders by them
            raise UnruledRender(f"{layout.name}: {rel} was measured with other owner's namespaces than the {graph} "
                                "render uses; remeasure the sizes and recompute the fit")
        if unmeasured:
            raise UnruledRender(f"{layout.name}: {rel} did not measure the rendered {graph} sources {unmeasured}; "
                                "remeasure the sizes over every arm and recompute the fit")


def _committed_json(layout: StoreLayout, root: Path, rel: str | None, sha: str | None, what: str) -> dict:
    """The JSON at ``root``/``rel``, refused unless it has the recorded sha256 and is committed (tracked and unmodified
    at HEAD of its repository)."""
    path = root / str(rel)
    if not rel or not path.is_file():
        raise UnruledRender(f"{layout.name}: {what} ({rel}) is missing")
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        raise UnruledRender(f"{layout.name}: {what} ({rel}) is not the file recorded (sha256 {str(sha)[:16]}...)")
    why = _committed(path, root)
    if why is not None:
        raise UnruledRender(f"{layout.name}: {what} ({rel}) must be committed: {why}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_ruled_render(renders_dir: Path, name: str) -> tuple[str, dict]:
    """The text and manifest of ``<renders_dir>/<name>.txt``, refused unless the manifest says its budget was ruled
    and the text is the one the manifest hashes."""
    manifest = json.loads((Path(renders_dir) / f"{name}.manifest.json").read_text(encoding="utf-8"))
    if manifest.get("budget_ruled") is not True:
        raise UnruledRender(f"{renders_dir}/{name}: rendered at {manifest.get('budget_tokens')} tokens, a budget no "
                            "ruling gives (budget_ruled is not true); rebuild it at the graph's ruled budget")
    text = (Path(renders_dir) / f"{name}.txt").read_text(encoding="utf-8")
    if hashlib.sha256(text.encode()).hexdigest() != manifest.get("text_sha256"):
        raise ValueError(f"{renders_dir}/{name}.txt is not the text its manifest's text_sha256 names")
    return text, manifest
