#!/usr/bin/env python3
"""One arm of the Gateway mitigation A/B (results/gateway-mitigations-20261020/AB-PLAN.md), run by the pilot runner.

    uv run python scripts/run_gateway_ab.py --arms          # every arm's base, route settings and caps; no request
    uv run python scripts/run_gateway_ab.py <run_pilot.py arguments> --models <arm or table model>

An arm is a model of the runner's table (scripts/run_pilot.py ``MODELS``) with only the named route settings and caps
changed; its key and every other setting are its base model's (tests/test_run_gateway_ab.py). The runner's table itself is
not touched, so no grid, manifest or pin sees an arm: arms are dev diagnostics, each run in its own scratch OUT
(a record's attempt key holds the route model, which an arm shares with its base), never pooled. A control arm is the
base model's own name run through this same script, so both arms run the same code. Which arm qualifies for adoption is
decided on the malfunction rate it targets, never on accuracy (change-log entry 113); adopting one is a change of the
model's MODELS entry with a recorded pin move, not this script.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _load_run_pilot():
    """The runner: the module already loaded under that name (the tests load it so), else scripts/run_pilot.py."""
    if "run_pilot" in sys.modules:
        return sys.modules["run_pilot"]
    spec = importlib.util.spec_from_file_location("run_pilot", Path(__file__).resolve().parent / "run_pilot.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_pilot"] = module
    spec.loader.exec_module(module)
    return module


run_pilot = _load_run_pilot()

# arm -> (base model of the runner's table, route options changed, caps changed). An arm copies every other field of
# its base. The one live A/B the study lead ruled (design log, 21 Oct 00:09; results/gateway-mitigations-20261020/
# GLM-ECHO-PLAN.md): GLM with its reasoning echoed against GLM without, both at GLM's ruled caps of 240 turns and
# 7,200 s (change-log entry 111), whatever the table this runs on gives GLM, so both arms run the same caps; the echo
# goes under "reasoning", the name GLM's deployment returns (it renders both names: the other-models probe of 20 Oct).
GLM_CAPS = {"max_turns": 240, "max_wall_s": 7200.0}
ARMS: dict[str, tuple[str, dict[str, Any], dict[str, Any]]] = {
    "glm-5.3-flash@240": ("glm-5.3-flash", {}, GLM_CAPS),
    "glm-5.3-flash@240+echo-r": ("glm-5.3-flash", {"echo_reasoning": "reasoning"}, GLM_CAPS),
}
LISTED_GRAPHS = ("uniprot-sp-2026_03", "rhea-142")


def arm_spec(base: Any, options: dict[str, Any], caps: dict[str, Any] | None = None) -> Any:
    """``base`` (a run_pilot.ModelSpec) with a route rebuilt from its own and ``options`` (OpenAICompatRoute
    .with_options, so the route's checks run again) and ``caps`` (ModelSpec fields, checked again by its
    __post_init__); every other field is the base's."""
    if not options:
        return dataclasses.replace(base, **(caps or {}))
    return dataclasses.replace(base, factory=lambda: base.factory().with_options(**options), **(caps or {}))


def arm_models(models: dict[str, Any]) -> dict[str, Any]:
    """Every arm of ``ARMS`` built on ``models`` (the runner's table)."""
    return {name: arm_spec(models[base], options, caps) for name, (base, options, caps) in ARMS.items()}


def listing(models: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for name, spec in arm_models(models).items():
        route = spec.factory()
        out[name] = {"base": ARMS[name][0], "route_model": route.model, "route_settings": route.route_info()["settings"],
                     "caps": {g: run_pilot.model_caps(spec, g) for g in LISTED_GRAPHS}}
    return out


def shared_route_models(names: list[str], models: dict[str, Any]) -> dict[str, list[str]]:
    """The route models two or more of ``names`` (models of ``models``) share, each with those names. An attempt's key
    holds its route model (run_pilot.attempt_key), which an arm shares with its base and with every other arm of it, so
    in one runner process one's claim would skip the other's attempt of the same question and repetition (External review
    of a2ea65c, item 5): each arm runs in a process, and an OUT, of its own."""
    by_route: dict[str, list[str]] = {}
    for name in names:
        if name in models:
            by_route.setdefault(models[name].factory().model, []).append(name)
    return {route: shared for route, shared in by_route.items() if len(shared) > 1}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["--arms"]:
        print(json.dumps(listing(run_pilot.MODELS), indent=1, sort_keys=True))
        return 0
    models = {**run_pilot.MODELS, **arm_models(run_pilot.MODELS)}
    # read as the runner reads them (its own parser, abbreviations included), so no spelling passes one and not the other
    args = run_pilot.parse_args(argv, models)
    if args.grid_manifest is not None:
        print("refused: the arms are dev diagnostics, never grid models, and a grid's models come from its manifest; "
              "run an arm with --models and an --out of its own", file=sys.stderr)
        return 2
    shared = shared_route_models(args.models or [], models)
    if shared:
        for route, names in shared.items():
            print(f"refused: {', '.join(names)} share the route model {route!r}, so their attempts' keys collide in "
                  f"one runner process; run each in a process and an --out of its own", file=sys.stderr)
        return 2
    return run_pilot.run(argv, models=models)


if __name__ == "__main__":
    sys.exit(main())
