"""Store metadata as it may be written to run records and pilot manifests.

The caller describes the store with a free-form dict (``--store-json``). Before
it is written anywhere it passes ``sanitize_store``:

* only the keys in ``STORE_KEYS`` are kept; any other key is dropped, and
  ``dropped_keys`` records how many were dropped and the names that match
  ``^[A-Za-z0-9_.-]{1,64}$`` (the redaction marker may stand inside a name; any
  other name is listed as ``<unprintable>``);
* a location (``endpoint``, ``endpoint_label``, ``url``) is reduced to scheme,
  host, port and path: user info, query and fragment are removed;
* every other URL, in a value, a list or a dict key, is an identifier (a graph
  IRI, which is not a credential): it is kept exactly, except that its user info
  is removed, so ``g?v=1`` and ``g?v=2`` stay distinct everywhere; dict keys are
  never otherwise rewritten, and two keys that become the same once user info is
  removed are refused (ValueError) rather than merged;
* at any depth, the value under a secret-named key is replaced by
  ``[redacted]``. A key is secret-named when one of its words (split at
  punctuation and camelCase) is password, passwd, pwd, secret, token, key, auth,
  authorization, credential(s), cookie or bearer, or ends in token, secret,
  password, passwd or key (``apiKey``, ``x-api-key``, ``accessToken``). Keys that
  are IRIs (graph names) are never treated as secret-named, so a graph called
  ``.../keywords`` or ``.../auth/...`` keeps its value;
* in any other (non-URL) string value, a secret-looking assignment
  (``token=...``, ``password: ...``, ``Bearer ...``) has its value replaced by
  ``[redacted]``.

Sanitising is idempotent: an existing ``dropped_keys`` record (or the plain list an
earlier version wrote) is carried over, so a stored, already sanitised description
sanitises to itself.

The record writer's environment-value redaction stays as a final safeguard over
the whole record and manifest; this removes credentials no environment
variable names.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

STORE_KEYS = frozenset({
    "name", "store", "engine", "version", "image", "image_digest", "identity", "graphs", "settings",
    "server_query_timeout", "default_graph_distinct_triples", "endpoint", "endpoint_label", "url", "description",
})
LOCATION_KEYS = frozenset({"endpoint", "endpoint_label", "url"})
REDACTED = "[redacted]"
UNPRINTABLE = "<unprintable>"
# The redaction marker may stand inside a name (a key that held a secret), so that a
# sanitised, then redacted, description sanitises to itself.
_PRINTABLE_NAME = re.compile(r"(?:[A-Za-z0-9_.-]|\[redacted\])+")

_SECRET_WORDS = frozenset({"password", "passwd", "pwd", "secret", "token", "key", "auth", "authorization",
                           "credential", "credentials", "cookie", "bearer"})
_SECRET_ENDINGS = ("token", "secret", "password", "passwd", "key")
_CAMEL = re.compile(r"([a-z0-9])([A-Z])")
_ASSIGNMENT = re.compile(
    r"(?i)\b(password|passwd|pwd|secret|token|key|api[_-]?key|access[_-]?key|auth|authorization)\b(\s*[=:]\s*)"
    r"([^\s,;&'\"]+)")
_BEARER = re.compile(r"(?i)\b(bearer)(\s+)([^\s,;&'\"]+)")


def is_secret_key(key: str) -> bool:
    """Whether a dict key names a secret (see the module docstring); IRI keys never do."""
    if "://" in key:
        return False
    words = [w for w in re.split(r"[^a-z0-9]+", _CAMEL.sub(r"\1_\2", key).lower()) if w]
    return any(w in _SECRET_WORDS or w.endswith(_SECRET_ENDINGS) for w in words)


def redact_assignments(text: str) -> str:
    text = _ASSIGNMENT.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", text)
    return _BEARER.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", text)


def public_url(url: str, *, keep_fragment: bool = False) -> str:
    """The URL without user info and query (and fragment unless kept); non-URLs unchanged."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if not parts.scheme or not parts.netloc:
        return url
    host = parts.hostname or ""
    if ":" in host:  # an IPv6 literal
        host = f"[{host}]"
    try:
        port = parts.port
    except ValueError:
        port = None
    if port:
        host = f"{host}:{port}"
    return urlunsplit((parts.scheme, host, parts.path, "", parts.fragment if keep_fragment else ""))


def strip_userinfo(url: str) -> str:
    """The URL without user info; everything else (path, query, fragment) unchanged."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if not parts.scheme or "@" not in parts.netloc:
        return url
    return urlunsplit((parts.scheme, parts.netloc.rsplit("@", 1)[1], parts.path, parts.query, parts.fragment))


def _key(key: Any) -> Any:
    if isinstance(key, str) and key.lower().startswith(("http://", "https://")):
        return strip_userinfo(key)
    return key


def _clean(value: Any) -> Any:
    if isinstance(value, str):
        if value.lower().startswith(("http://", "https://")):
            return strip_userinfo(value)  # an identifier: kept exactly but for its user info
        return redact_assignments(value)
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            key = _key(k)
            if key in out:
                raise ValueError(f"store metadata keys collide once user info is removed: {key!r}")
            out[key] = REDACTED if isinstance(k, str) and is_secret_key(k) else _clean(v)
        return out
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def sanitize_store(store: dict | None) -> dict:
    """The store description with only allowed keys and no credentials in URLs."""
    out: dict[str, Any] = {}
    count = 0
    names: list[str] = []
    for key, value in (store or {}).items():
        if key == "dropped_keys" and isinstance(value, dict):  # already sanitised once
            count += int(value.get("count") or 0)
            names.extend(str(n) for n in value.get("names") or [])
        elif key == "dropped_keys" and isinstance(value, list):  # the form an earlier version wrote
            count += len(value)
            names.extend(str(n) for n in value)
        elif key not in STORE_KEYS:
            count += 1
            names.append(str(key))
        elif key in LOCATION_KEYS and isinstance(value, str):
            out[key] = public_url(value)
        else:
            out[key] = _clean(value)
    if count:
        shown = sorted(n if len(n) <= 64 and _PRINTABLE_NAME.fullmatch(n) else UNPRINTABLE for n in names)
        out["dropped_keys"] = {"count": count, "names": shown}
    return out
