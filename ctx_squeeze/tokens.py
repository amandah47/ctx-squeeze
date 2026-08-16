"""Heuristic token estimation with no third-party dependencies.

This module deliberately does *not* implement a real BPE tokenizer. It walks the
text once and scores each run of characters with a weight that approximates how
byte-pair encoders behave in practice:

* ASCII words       ~ 4 characters per token (minimum one token per word)
* digit runs        ~ 3 characters per token (minimum one token per run)
* CJK characters    ~ 1 token per character
* newlines          ~ 0.5 tokens each (runs of newlines merge)
* other symbols     ~ 0.6 tokens each
* spaces/tabs       free (they attach to the following token)

On ordinary English prose the result lands within roughly 10% of the count a
BPE tokenizer reports, which is accurate enough for budgeting decisions and
cheap enough to run on every paragraph of a large document.
"""

import math
import re

__all__ = [
    "estimate_tokens",
    "fits_budget",
    "truncate_to_tokens",
    "CHARS_PER_WORD_TOKEN",
    "CHARS_PER_NUMBER_TOKEN",
    "TOKENS_PER_CJK_CHAR",
    "TOKENS_PER_NEWLINE",
    "TOKENS_PER_SYMBOL",
]

CHARS_PER_WORD_TOKEN = 4.0
CHARS_PER_NUMBER_TOKEN = 3.0
TOKENS_PER_CJK_CHAR = 1.0
TOKENS_PER_NEWLINE = 0.5
TOKENS_PER_SYMBOL = 0.6

_SCANNER = re.compile(
    r"(?P<newline>\n+)"
    r"|(?P<space>[ \t\r\f\v]+)"
    r"|(?P<word>[A-Za-z]+(?:['’][A-Za-z]+)*)"
    r"|(?P<number>[0-9]+)"
    r"|(?P<cjk>[぀-ヿ㐀-䶿一-鿿가-힯]+)"
    r"|(?P<other>.)",
    re.DOTALL,
)


def estimate_tokens(text):
    """Return an estimated token count for ``text``.

    The result is always a non-negative integer, and is zero only for the empty
    string or a string made purely of horizontal whitespace.
    """
    if not text:
        return 0
    total = 0.0
    for match in _SCANNER.finditer(text):
        kind = match.lastgroup
        length = match.end() - match.start()
        if kind == "word":
            total += max(1.0, length / CHARS_PER_WORD_TOKEN)
        elif kind == "number":
            total += max(1.0, length / CHARS_PER_NUMBER_TOKEN)
        elif kind == "cjk":
            total += length * TOKENS_PER_CJK_CHAR
        elif kind == "newline":
            total += length * TOKENS_PER_NEWLINE
        elif kind == "other":
            total += length * TOKENS_PER_SYMBOL
    return int(math.ceil(total))


def fits_budget(text, budget):
    """True when ``text`` is estimated to fit inside ``budget`` tokens."""
    return estimate_tokens(text) <= budget


def truncate_to_tokens(text, budget, suffix=""):
    """Hard-truncate ``text`` so the estimate is at most ``budget`` tokens.

    This is the last-resort path used when a single indivisible block is still
    larger than the whole budget. ``suffix`` (if given) is counted against the
    budget and appended to the result.
    """
    if budget <= 0:
        return ""
    if fits_budget(text, budget):
        return text
    room = budget - estimate_tokens(suffix)
    if room <= 0:
        return ""
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if estimate_tokens(text[:mid]) <= room:
            low = mid
        else:
            high = mid - 1
    return text[:low].rstrip() + suffix
