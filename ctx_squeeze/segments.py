"""Split raw text into the units that compaction operates on.

A segment is a paragraph, a heading, a list run, or a whole fenced code block.
Code fences are kept intact: cutting a code block in half produces text that is
worse than useless to a model, so the compactor treats them as atoms.
"""

from .tokens import estimate_tokens

__all__ = ["Segment", "split_segments", "join_segments", "FENCES"]

FENCES = ("```", "~~~")


class Segment(object):
    """One indivisible chunk of the source text."""

    __slots__ = ("text", "start_line", "end_line", "kind", "tokens", "index")

    def __init__(self, text, start_line, end_line, kind="text", index=0):
        self.text = text
        self.start_line = start_line
        self.end_line = end_line
        self.kind = kind
        self.index = index
        self.tokens = estimate_tokens(text)

    @property
    def is_code(self):
        return self.kind == "code"

    def __repr__(self):
        return "Segment(index=%d, kind=%s, lines=%d-%d, tokens=%d)" % (
            self.index,
            self.kind,
            self.start_line,
            self.end_line,
            self.tokens,
        )

    def __eq__(self, other):
        if not isinstance(other, Segment):
            return NotImplemented
        return (
            self.text == other.text
            and self.start_line == other.start_line
            and self.end_line == other.end_line
            and self.kind == other.kind
            and self.index == other.index
        )

    def __hash__(self):
        return hash((self.text, self.start_line, self.kind, self.index))


def _fence_marker(stripped_line):
    for fence in FENCES:
        if stripped_line.startswith(fence):
            return fence
    return None


def split_segments(text):
    """Split ``text`` into a list of :class:`Segment` in document order.

    Blank lines separate segments. Fenced code blocks survive whole, including
    any blank lines inside them. An unterminated fence runs to end of input.
    """
    if not text or not text.strip():
        return []

    lines = text.split("\n")
    segments = []
    buffer = []
    buffer_start = 0
    fence = None

    def flush(last_index, kind):
        if not buffer:
            return
        body = "\n".join(buffer).strip("\n")
        if body.strip():
            segments.append(
                Segment(
                    text=body,
                    start_line=buffer_start + 1,
                    end_line=last_index + 1,
                    kind=kind,
                    index=len(segments),
                )
            )
        del buffer[:]

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if fence is None:
            marker = _fence_marker(stripped)
            if marker is not None:
                flush(idx - 1, "text")
                fence = marker
                buffer.append(line)
                buffer_start = idx
                continue
            if not stripped:
                flush(idx - 1, "text")
                continue
            if not buffer:
                buffer_start = idx
            buffer.append(line)
        else:
            buffer.append(line)
            if _fence_marker(stripped) == fence:
                flush(idx, "code")
                fence = None

    flush(len(lines) - 1, "code" if fence is not None else "text")
    return segments


def join_segments(segments, separator="\n\n"):
    """Re-assemble segment text with ``separator`` between adjacent chunks."""
    return separator.join(segment.text for segment in segments)
