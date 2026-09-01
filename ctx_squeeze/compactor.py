"""The compaction pipeline: strategies that pick which segments survive a budget.

``squeeze`` splits text into segments, runs a comma-separated chain of named
stages over them, and reassembles whatever survives. A stage is either a
filter (``dedupe`` narrows the candidates without looking at the budget) or a
selector (``score``, ``head-tail`` pick a subset that fits the budget). The
docstring in tokens.py explains why nothing here ever truncates a segment:
half a code block is worse than no code block, so a segment that cannot fit
is dropped whole rather than cut.

Dropped runs are replaced by a single "[N segments elided]" marker, and that
marker itself costs tokens. To keep the "always fits budget" promise honest,
the result is checked once fully assembled - markers included - and trimmed
further if the markers pushed it over.
"""

from .dedupe import dedupe_segments
from .scoring import score_segments, select_by_score
from .segments import Segment, join_segments, split_segments
from .tokens import estimate_tokens

__all__ = ["STRATEGIES", "SqueezeResult", "squeeze"]


def _dedupe_stage(segments, budget, jaccard_threshold=0.8, shingle_size=5, **_ignored):
    kept = dedupe_segments(segments, threshold=jaccard_threshold, shingle_size=shingle_size)
    dropped = len(segments) - len(kept)
    note = "dedupe dropped %d near-duplicate segment(s)" % dropped if dropped else None
    return kept, note


def _score_stage(segments, budget, **_ignored):
    return select_by_score(segments, budget), None


def _head_tail_stage(segments, budget, head_ratio=0.5, **_ignored):
    if not segments or budget <= 0:
        return [], None

    head_ratio = min(1.0, max(0.0, head_ratio))
    head_budget = int(round(budget * head_ratio))
    tail_budget = budget - head_budget

    head = []
    spent = 0
    for segment in segments:
        if spent + segment.tokens > head_budget:
            break
        head.append(segment)
        spent += segment.tokens

    head_indices = {segment.index for segment in head}
    tail = []
    spent = 0
    for segment in reversed(segments):
        if segment.index in head_indices:
            break
        if spent + segment.tokens > tail_budget:
            break
        tail.append(segment)
        spent += segment.tokens
    tail.reverse()

    return head + tail, None


STRATEGIES = {
    "dedupe": _dedupe_stage,
    "score": _score_stage,
    "head-tail": _head_tail_stage,
}


def _elision_marker(count):
    text = "[%d segment%s elided]" % (count, "" if count == 1 else "s")
    return Segment(text=text, start_line=0, end_line=0, kind="marker", index=-1)


def _assemble(original_segments, kept_segments, marker):
    kept_indices = {segment.index for segment in kept_segments}
    pieces = []
    elided = 0
    for segment in original_segments:
        if segment.index in kept_indices:
            if elided and marker:
                pieces.append(_elision_marker(elided))
            elided = 0
            pieces.append(segment)
        else:
            elided += 1
    if elided and marker:
        pieces.append(_elision_marker(elided))
    return pieces


def _fits(pieces, budget):
    return sum(piece.tokens for piece in pieces) <= budget


def _trim_to_budget(original_segments, kept_segments, budget, marker):
    """Drop the lowest-scoring kept segments until the assembled text (markers
    included) fits ``budget``. Returns the final pieces and how many segments
    this step removed on top of whatever the strategy already chose.
    """
    kept_segments = list(kept_segments)
    pieces = _assemble(original_segments, kept_segments, marker)
    if _fits(pieces, budget) or not kept_segments:
        return pieces, 0

    scores = dict(zip((segment.index for segment in kept_segments), score_segments(kept_segments)))
    trimmed = 0
    while kept_segments and not _fits(pieces, budget):
        worst = min(kept_segments, key=lambda segment: (scores[segment.index], -segment.tokens))
        kept_segments.remove(worst)
        trimmed += 1
        pieces = _assemble(original_segments, kept_segments, marker)

    if not _fits(pieces, budget):
        # Nothing left but the marker itself, and it still doesn't fit: drop it too.
        pieces = _assemble(original_segments, kept_segments, marker=False)

    return pieces, trimmed


class SqueezeResult(object):
    """The outcome of :func:`squeeze`."""

    __slots__ = ("text", "original_tokens", "final_tokens", "segments_in", "segments_out", "notes")

    def __init__(self, text, original_tokens, final_tokens, segments_in, segments_out, notes):
        self.text = text
        self.original_tokens = original_tokens
        self.final_tokens = final_tokens
        self.segments_in = segments_in
        self.segments_out = segments_out
        self.notes = notes

    def __repr__(self):
        return "SqueezeResult(kept=%d of %d, tokens=%d -> %d)" % (
            self.segments_out,
            self.segments_in,
            self.original_tokens,
            self.final_tokens,
        )


def squeeze(text, budget, strategy="score", head_ratio=0.5, jaccard_threshold=0.8, shingle_size=5, marker=True):
    """Compact ``text`` to fit ``budget`` estimated tokens.

    ``strategy`` is a comma-separated pipeline of stage names from
    :data:`STRATEGIES`, applied left to right: a filter stage such as
    ``"dedupe"`` narrows the candidate segments before a selector stage such
    as ``"score"`` or ``"head-tail"`` picks a subset that fits the budget.
    Dropped runs of segments are replaced by a single elision marker unless
    ``marker`` is False. The result never exceeds ``budget`` for a
    non-negative budget, markers included.
    """
    stage_names = [name.strip() for name in strategy.split(",") if name.strip()]
    if not stage_names:
        raise ValueError("strategy must name at least one stage")
    for name in stage_names:
        if name not in STRATEGIES:
            raise ValueError("unknown strategy %r (choices: %s)" % (name, ", ".join(sorted(STRATEGIES))))

    budget = max(0, budget)
    original_segments = split_segments(text)
    original_tokens = estimate_tokens(text)
    segments_in = len(original_segments)

    if not original_segments or original_tokens <= budget:
        return SqueezeResult(
            text=text,
            original_tokens=original_tokens,
            final_tokens=original_tokens,
            segments_in=segments_in,
            segments_out=segments_in,
            notes=[],
        )

    notes = []
    segments = original_segments
    for name in stage_names:
        segments, note = STRATEGIES[name](
            segments,
            budget,
            head_ratio=head_ratio,
            jaccard_threshold=jaccard_threshold,
            shingle_size=shingle_size,
        )
        if note:
            notes.append(note)

    pieces, trimmed = _trim_to_budget(original_segments, segments, budget, marker)
    if trimmed:
        notes.append("dropped %d additional segment(s) to keep elision markers within budget" % trimmed)

    final_text = join_segments(pieces) if pieces else ""
    kept_count = sum(1 for piece in pieces if piece.kind != "marker")

    return SqueezeResult(
        text=final_text,
        original_tokens=original_tokens,
        final_tokens=estimate_tokens(final_text),
        segments_in=segments_in,
        segments_out=kept_count,
        notes=notes,
    )
