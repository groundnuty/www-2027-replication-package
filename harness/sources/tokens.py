"""The one tokenizer that measures render budgets.

tiktoken's ``o200k_base`` BPE, offline and deterministic once its file is
cached. Budgets are therefore in o200k tokens, not in Claude or Qwen tokens;
the point is that the budget is the same, measured the same way, for every
source. The BPE file is checked against the sha256 below before first use. If
it cannot be loaded (no network and no cache: set ``TIKTOKEN_CACHE_DIR`` to a
pre-filled directory on HPC or CI) this fails; it never falls back to another
count.
"""

from __future__ import annotations

import importlib.metadata
from functools import lru_cache

import tiktoken
import tiktoken.load

ENCODING = "o200k_base"
BPE_URL = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
BPE_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    tiktoken.load.read_file_cached(BPE_URL, BPE_SHA256)  # raises on a missing or altered file
    return tiktoken.get_encoding(ENCODING)


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text, disallowed_special=()))


def tokenizer_info() -> dict:
    return {
        "library": "tiktoken",
        "version": importlib.metadata.version("tiktoken"),
        "encoding": ENCODING,
        "bpe_sha256": BPE_SHA256,
        "unit": "o200k_base tokens (not Claude or Qwen tokens)",
    }
