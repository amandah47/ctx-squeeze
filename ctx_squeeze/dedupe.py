"""Near-duplicate detection for segments, using shingling and Jaccard similarity.

Agent transcripts repeat themselves: the same file read three times, the same
traceback echoed after each retry, a tool result restated in the next message.
Those repeats rarely match byte-for-byte (a timestamp or line number differs),
so exact-match dedup misses most of them. Shingling turns each segment into a
set of overlapping word n-grams, and Jaccard similarity on those sets tolerates
the small differences while still catching the wholesale repeat.
"""

import re

__all__ = ["shingles", "jaccard", "find_near_duplicates", "dedupe_segments"]

_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def shingles(text, size=5):
    """Return the set of overlapping word n-gram shingles in ``text``.

    Words are lowercased before shingling so casing differences don't count
    against similarity. A text with fewer words than ``size`` yields a single
    shingle covering everything it has, so short segments can still be
    compared. An empty or all-punctuation text yields an empty set.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    words = _WORD.findall(text.lower())
    if not words:
        return set()
    if len(words) <= size:
        return {tuple(words)}
    return {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}


def jaccard(a, b):
    """Jaccard similarity between two shingle sets: ``|a & b| / |a | b|``.

    Two empty sets are treated as identical (similarity 1.0); one empty and
    one non-empty set have nothing in common (similarity 0.0).
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union


def find_near_duplicates(segments, threshold=0.8, shingle_size=5):
    """Return ``(i, j)`` index pairs, ``i < j``, of segments that are near-duplicates.

    Comparison is pairwise over ``segments`` as given; indices refer to
    position in that list, not to ``Segment.index``.
    """
    sets = [shingles(segment.text, shingle_size) for segment in segments]
    pairs = []
    for i in range(len(segments)):
        if not sets[i]:
            continue
        for j in range(i + 1, len(segments)):
            if not sets[j]:
                continue
            if jaccard(sets[i], sets[j]) >= threshold:
                pairs.append((i, j))
    return pairs


def dedupe_segments(segments, threshold=0.8, shingle_size=5):
    """Return a copy of ``segments`` with near-duplicates removed.

    Segments are kept in their original order. For each near-duplicate group
    the first occurrence is kept and later ones are dropped, so an earlier
    read of a file survives over a later, redundant one.
    """
    kept = []
    kept_sets = []
    for segment in segments:
        segment_shingles = shingles(segment.text, shingle_size)
        duplicate = False
        if segment_shingles:
            for other_shingles in kept_sets:
                if other_shingles and jaccard(segment_shingles, other_shingles) >= threshold:
                    duplicate = True
                    break
        if not duplicate:
            kept.append(segment)
            kept_sets.append(segment_shingles)
    return kept
