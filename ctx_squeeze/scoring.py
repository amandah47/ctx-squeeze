"""Extractive scoring for segments, using TF-IDF keyword density.

A segment that reuses the document's own distinctive vocabulary is more likely
to carry its point than one that is mostly connective prose. Scoring each
segment by TF-IDF and then selecting greedily by score gives an extractive
excerpt that favors keyword-dense text without a model call.
"""

import math
import re

__all__ = ["score_segments", "select_by_score"]

_WORD = re.compile(r"[^\W_]+", re.UNICODE)

# Common English function words carry no keyword signal and would otherwise
# dominate every segment's term frequency regardless of what it's actually about.
_STOPWORDS = frozenset(
    """
    a an and are as at be but by for from had has have he her his i in is it
    its of on or our that the their them these they this to was we were will
    with you your
    """.split()
)


def _words(text):
    return [word for word in _WORD.findall(text.lower()) if word not in _STOPWORDS]


def score_segments(segments):
    """Return a list of TF-IDF density scores, one per segment, in input order.

    Each segment's score is the sum of its terms' TF-IDF weights divided by its
    word count, so a short, keyword-dense segment (a heading that repeats the
    document's title terms) scores competitively against a long one. A segment
    with no scorable words (empty, or stopwords only) gets a score of 0.0.
    """
    word_lists = [_words(segment.text) for segment in segments]

    document_frequency = {}
    for words in word_lists:
        for word in set(words):
            document_frequency[word] = document_frequency.get(word, 0) + 1

    document_count = len(segments)
    scores = []
    for words in word_lists:
        if not words:
            scores.append(0.0)
            continue
        term_frequency = {}
        for word in words:
            term_frequency[word] = term_frequency.get(word, 0) + 1
        density = 0.0
        for word, count in term_frequency.items():
            idf = math.log((document_count + 1) / (document_frequency[word] + 1)) + 1.0
            density += (count / len(words)) * idf
        scores.append(density)
    return scores


def select_by_score(segments, budget):
    """Greedily select segments by descending score until ``budget`` is spent.

    Segments are considered highest score first; a segment that would push the
    running total over budget is skipped rather than stopping the scan, so a
    later, cheaper, high-scoring segment still gets a chance. Ties break by
    original position, earlier first. The return value keeps selected segments
    in document order, not selection order, so the result reads like an excerpt.
    """
    if budget <= 0 or not segments:
        return []

    scores = score_segments(segments)
    order = sorted(range(len(segments)), key=lambda i: (-scores[i], i))

    selected = set()
    spent = 0
    for i in order:
        cost = segments[i].tokens
        if spent + cost > budget:
            continue
        selected.add(i)
        spent += cost

    return [segments[i] for i in sorted(selected)]
