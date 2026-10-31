"""Shared cl100k tokenizer for every token-size measurement (Task 2.3 ruling 2).

All three chunkers measure ``size_tokens``/``overlap_tokens``/``min_tokens``/``max_tokens``
with the same pinned tiktoken encoding (``cl100k_base``) so the unit means the same thing
across strategies. Before first use the tiktoken BPE cache is pointed at the repo-relative
``.tiktoken_cache/`` directory (gitignored), keeping the ~1.6MB vocabulary download on the
project drive. tiktoken 0.14 exposes no char-offset API, so token boundaries are mapped to
char offsets through each token's UTF-8 byte length.
"""

from __future__ import annotations

import os
from pathlib import Path

import tiktoken

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CACHE_DIR = _REPO_ROOT / ".tiktoken_cache"
_ENCODING_NAME = "cl100k_base"


def get_tokenizer() -> tiktoken.Encoding:
    """Return the shared cl100k_base encoding, downloading its BPE into ``.tiktoken_cache`` once."""
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(_CACHE_DIR))
    return tiktoken.get_encoding(_ENCODING_NAME)


def count_tokens(text: str) -> int:
    """Token count of ``text`` under the shared cl100k_base encoding."""
    return len(get_tokenizer().encode_ordinary(text))


def token_char_offsets(text: str) -> list[int]:
    """Char offsets of every token boundary in ``text``; length is ``n_tokens + 1``.

    The returned list ``o`` satisfies ``text[o[i]:o[i + 1]] == decode(tokens[i])`` for all
    valid ``i``, so slicing the document by these offsets reproduces token windows exactly.
    Boundaries are computed from the cumulative UTF-8 byte length of each token; a byte
    offset that lands mid-character (possible in principle for byte-level BPE) maps to the
    earlier char boundary via ``errors="ignore"`` decoding, which keeps slices exact.
    """
    enc = get_tokenizer()
    tokens = enc.encode_ordinary(text)
    byte_to_char: dict[int, int] = {0: 0}
    byte_pos = 0
    for index, char in enumerate(text):
        byte_pos += len(char.encode("utf-8"))
        byte_to_char[byte_pos] = index + 1
    offsets = [0]
    total_bytes = 0
    for token in tokens:
        total_bytes += len(enc.decode_single_token_bytes(token))
        if total_bytes in byte_to_char:
            offsets.append(byte_to_char[total_bytes])
        else:
            raw = text.encode("utf-8")[:total_bytes]
            offsets.append(len(raw.decode("utf-8", errors="ignore")))
    return offsets
