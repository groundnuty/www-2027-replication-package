"""IRI shapes of replacement constants (design ruling (c) in the design log, change-log entry 63).

The constants check (results/constants-check-20261019/shapes.py) classified the bank's replacement
constants and the agents' wrong constants with one rule; the bank's ``internal_node`` descriptor and
its replacement forms use that rule unchanged, quirks included (any run of twelve hex characters,
digits among them, counts as hashed), so the descriptor means what the check measured.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit

HEX = re.compile(r"[0-9A-Fa-f]{12,}")
FORMS = ("internal node", "2 other host", "3 other namespace", "4 same namespace, other kind",
         "5 same namespace and kind")


def split(iri: str) -> dict:
    base, frag = (iri.split("#", 1) + [None])[:2] if "#" in iri else (iri, None)
    last = base.rstrip("/").rsplit("/", 1)[-1]
    entity_frag = frag is not None and bool(re.search(r"\d", last))  # a fragment on an entity IRI, not a vocabulary '#'
    if entity_frag:
        ns, local = base[: base.rfind("/") + 1], base[base.rfind("/") + 1:]
    else:
        cut = max(iri.rfind("/"), iri.rfind("#"))
        ns, local = iri[: cut + 1], iri[cut + 1:]
    return {"ns": ns, "local": local, "frag": frag if entity_frag else None, "host": urlsplit(iri).netloc}


def kind(local: str) -> str:
    if re.fullmatch(r"\d+", local):
        return "digits"
    if (re.fullmatch(r"[A-Z][A-Z0-9]{5,9}", local) and re.search(r"\d", local)
            and re.search(r"[A-Z].*[A-Z]|[A-Z]\d", local)
            and re.fullmatch(r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2}", local)):
        return "uniprot_accession"
    m = re.fullmatch(r"([A-Za-z]+[_:]?)(\d+)", local)
    if m:
        return "prefix_number:" + m.group(1)
    if re.fullmatch(r"[A-Za-z][A-Za-z_\-]*", local):
        return "word"
    if "." in local:
        return "dotted"
    return "other"


def flags(iri: str) -> dict:
    s = split(iri)
    return {"fragment": s["frag"] is not None, "pct": "%" in iri, "dotted": "." in s["local"],
            "hashed": bool(HEX.search(s["frag"] or s["local"])), "kind": kind(s["local"]), **s}


def category(iri: str, refs: list[str]) -> str:
    """Primary shape category of ``iri`` relative to reference IRIs: the first of fragment, other
    host, other namespace, same namespace with another local-name kind, or same namespace and kind."""
    f = flags(iri)
    if f["fragment"]:
        return "1 fragment"
    rs = [flags(r) for r in refs]
    if not any(r["host"] == f["host"] for r in rs):
        return "2 other host"
    if not any(r["ns"] == f["ns"] for r in rs):
        return "3 other namespace"
    if not any(r["ns"] == f["ns"] and r["kind"] == f["kind"] for r in rs):
        return "4 same namespace, other kind"
    return "5 same namespace and kind"


def internal(iri: str) -> bool:
    """A store-internal node rather than a named entity: a fragment on an entity IRI, a '#_'
    fragment (citation statements), or a SIP/MD5/hex-hashed local name or fragment."""
    s = split(iri)
    tail = iri.split("#", 1)[1] if "#" in iri else s["local"]
    return (s["frag"] is not None or "#_" in iri or bool(HEX.search(tail))
            or bool(re.match(r"(SIP|gene-MD5)", tail)))


def form(iri: str, refs: list[str]) -> str:
    """The check's form of a constant: an internal node, else its category against ``refs`` (the
    question's gold constants), as the check's final table counted them."""
    return "internal node" if internal(iri) else category(iri, refs)
