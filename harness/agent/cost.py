"""Token cost ESTIMATES per route, from a list-price table, and the cache hit rate.

These are estimates: list prices times the token counts the provider reported,
not an invoice. Anthropic prices are USD per million tokens from Anthropic's
price page (PRICE_SOURCE, retrieved 2026-10-18), each normalised usage count
priced separately (see providers.py for the disjoint convention). They assume
no ``inference_geo``: US-only inference would cost 1.1x, and no route asks for it.
Cache writes have two prices by lifetime: where the provider breaks writes down
by lifetime (``cache_write_5m_tokens`` / ``cache_write_1h_tokens`` in the
record's totals, Anthropic's ``usage.cache_creation`` per turn) each part is
priced at its own rate; any write not broken down is priced at the route's
``cache_ttl`` rate (5 minutes when the route does not say).

Claude Haiku 5.5 is priced by each request's prompt length: all its input
tokens (uncached input, cache reads and cache writes) up to 100,000 at the list
price, longer at the long-prompt price. A record's turns are its requests, so
``estimate_record_cost`` prices each turn on its own and sums them; counts
without per-turn usage may span requests of unknown lengths and are priced at
the long-prompt rate, an upper bound. A call made through the Message Batches
API (``batch=True``: a closed-book probe record of call mode "batch") is billed
at half the list price for every kind of token, cache reads and writes
included (the cache multipliers stack with the Batch discount).

Models served by HPCNet LLM Gateway cost nothing per token: compute there is
covered by a HPCNet grant.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

PRICE_SOURCE = ("Anthropic price page https://platform.claude.com/docs/en/about-claude/pricing, retrieved 2026-10-18 "
                "(list prices, USD per million tokens; no inference_geo, whose US-only inference costs 1.1x)")

# provider -> model -> USD per million tokens; for a model in LONG_PROMPT_PRICE_USD_PER_MTOK, the price of a
# request whose prompt is at most LONG_PROMPT_THRESHOLD tokens
LIST_PRICE_USD_PER_MTOK: dict[str, dict[str, dict[str, float]]] = {
    "anthropic": {
        "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_read": 0.10,
                              "cache_write_5m": 2.50, "cache_write_1h": 4.00},
        "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_read": 0.01,
                             "cache_write_5m": 0.125, "cache_write_1h": 0.20},
        "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_read": 0.20,
                            "cache_write_5m": 5.00, "cache_write_1h": 8.00},
    },
}
# provider -> model -> USD per million tokens of a request whose prompt (uncached input + cache reads + cache
# writes) is longer than LONG_PROMPT_THRESHOLD tokens; a model not listed costs the same at any length
LONG_PROMPT_THRESHOLD = 100_000
LONG_PROMPT_PRICE_USD_PER_MTOK: dict[str, dict[str, dict[str, float]]] = {
    "anthropic": {
        "claude-haiku-5-5": {"input": 0.50, "output": 2.50, "cache_read": 0.05,
                             "cache_write_5m": 0.625, "cache_write_1h": 1.00},
    },
}
BATCH_DISCOUNT = 0.5  # the Message Batches API bills every token at half the list price

GRANT_HOSTS = {"gateway.example": "HPCNet grant: no per-token charge on HPCNet LLM Gateway"}

_FREE = {"input": 0.0, "output": 0.0, "cache_read": 0.0, "cache_write_5m": 0.0, "cache_write_1h": 0.0}
_PROMPT_COUNTS = ("input_tokens", "cache_read_tokens", "cache_write_tokens")


def price_for(route: dict) -> tuple[dict[str, float] | None, str]:
    """(price per MTok by usage kind, basis) for a route_info, or (None, why not)."""
    provider = route.get("provider")
    model = route.get("model")
    host = urlsplit(route.get("base_url") or "").hostname or ""
    if provider == "openai-compatible" and host in GRANT_HOSTS:
        return dict(_FREE), GRANT_HOSTS[host]
    prices = LIST_PRICE_USD_PER_MTOK.get(provider or "", {}).get(model or "")
    if prices is None:
        return None, f"no list price for provider {provider!r} model {model!r}"
    return prices, PRICE_SOURCE


def long_prompt_price_for(route: dict) -> dict[str, float] | None:
    """The price per MTok of a request longer than LONG_PROMPT_THRESHOLD, for a model priced by prompt length."""
    return LONG_PROMPT_PRICE_USD_PER_MTOK.get(route.get("provider") or "", {}).get(route.get("model") or "")


def _route_ttl(route: dict) -> str:
    ttl = (route.get("settings") or {}).get("cache_ttl")
    return ttl if ttl in ("5m", "1h") else "5m"


def _usd(prices: dict[str, float], counts: dict[str, Any], ttl: str) -> tuple[float, list[str]]:
    """Unrounded USD of the counts at these prices, and the counts the provider did not report."""
    usd = 0.0
    unreported = []
    for count_key, price_key in (("input_tokens", "input"), ("output_tokens", "output"),
                                 ("cache_read_tokens", "cache_read")):
        count = counts.get(count_key)
        if count is None:
            unreported.append(count_key)
        else:
            usd += count / 1_000_000 * prices[price_key]
    writes = counts.get("cache_write_tokens")
    if writes is None:
        unreported.append("cache_write_tokens")
    else:
        by_ttl = {t: counts.get(f"cache_write_{t}_tokens") or 0 for t in ("5m", "1h")}
        by_ttl[ttl] += max(writes - sum(by_ttl.values()), 0)
        usd += sum(n / 1_000_000 * prices[f"cache_write_{t}"] for t, n in by_ttl.items())
    return usd, unreported


def estimate_cost(route: dict, totals: dict[str, Any], *, batch: bool = False) -> dict:
    """An estimate of what the token counts cost on this route (``batch``: through the Batches API).

    For a model priced by prompt length the counts may span requests of unknown lengths, so they are priced at
    the long-prompt rate, an upper bound; ``estimate_record_cost`` prices a record with per-turn usage turn by turn.
    """
    prices, basis = price_for(route)
    long_prices = long_prompt_price_for(route)
    if prices is not None and long_prices is not None:
        prices = long_prices
        basis = (f"{basis}; no per-request usage: priced at the rate for prompts over {LONG_PROMPT_THRESHOLD} "
                 "tokens (an upper bound)")
    if prices is not None and batch:
        prices = {k: v * BATCH_DISCOUNT for k, v in prices.items()}
        basis = f"{basis}; Message Batches API: {BATCH_DISCOUNT:g} of list price"
    if prices is None:
        return {"usd": None, "is_estimate": True, "basis": basis, "unreported": []}
    usd, unreported = _usd(prices, totals, _route_ttl(route))
    return {"usd": round(usd, 6), "is_estimate": True, "basis": basis, "unreported": unreported}


def _turn_counts(turn: dict) -> dict[str, Any]:
    """One turn's usage in the shape of the totals, writes broken down by lifetime as loop.token_totals does."""
    counts = {k: turn.get(k) for k in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")}
    creation = (turn.get("usage_raw") or {}).get("cache_creation")
    for ttl in ("5m", "1h"):
        n = creation.get(f"ephemeral_{ttl}_input_tokens") if isinstance(creation, dict) else None
        counts[f"cache_write_{ttl}_tokens"] = n if isinstance(n, int) else None
    return counts


def _estimate_turns_cost(route: dict, per_turn: list[dict], *, batch: bool) -> dict:
    """Each turn (one request) at the rate for its own prompt length; a prompt with an unreported count is long."""
    short_prices, basis = price_for(route)
    long_prices = long_prompt_price_for(route)
    if short_prices is None or long_prices is None:
        raise ValueError(f"no prompt-length prices for {route.get('provider')!r} {route.get('model')!r}")
    if batch:
        short_prices = {k: v * BATCH_DISCOUNT for k, v in short_prices.items()}
        long_prices = {k: v * BATCH_DISCOUNT for k, v in long_prices.items()}
    ttl = _route_ttl(route)
    usd = 0.0
    unreported: list[str] = []
    n_long = 0
    for turn in per_turn:
        counts = _turn_counts(turn)
        prompt = [counts[k] for k in _PROMPT_COUNTS]
        is_long = None in prompt or sum(prompt) > LONG_PROMPT_THRESHOLD
        n_long += is_long
        part, missing = _usd(long_prices if is_long else short_prices, counts, ttl)
        usd += part
        unreported += [k for k in missing if k not in unreported]
    basis = (f"{basis}; each request priced on its own prompt length: {len(per_turn) - n_long} at most "
             f"{LONG_PROMPT_THRESHOLD} tokens, {n_long} over")
    if batch:
        basis = f"{basis}; Message Batches API: {BATCH_DISCOUNT:g} of list price"
    return {"usd": round(usd, 6), "is_estimate": True, "basis": basis, "unreported": unreported}


def estimate_record_cost(record: Any) -> dict:
    """``estimate_cost`` for a RunRecord (or its JSON form). A model priced by prompt length is priced turn by
    turn from the record's per-turn usage; a record without it, on its totals at the long-prompt rate."""
    data = record.to_json() if hasattr(record, "to_json") else record
    batch = (data.get("call") or {}).get("mode") == "batch"  # a closed-book probe record (harness/probe)
    per_turn = data["tokens"].get("per_turn")
    if per_turn and long_prompt_price_for(data["route"]) is not None:
        return _estimate_turns_cost(data["route"], per_turn, batch=batch)
    return estimate_cost(data["route"], data["tokens"]["totals"], batch=batch)


def cache_hit_rate(totals: dict[str, Any]) -> float | None:
    """cache_read / (input + cache_read + cache_write) over an attempt; None when the
    provider did not report cache reads or no input was counted."""
    read = totals.get("cache_read_tokens")
    if read is None:
        return None
    total = sum(totals.get(k) or 0 for k in ("input_tokens", "cache_read_tokens", "cache_write_tokens"))
    return read / total if total else None
