from ctx_squeeze.dedupe import dedupe_segments, find_near_duplicates, jaccard, shingles
from ctx_squeeze.segments import Segment


def make_segment(text, index=0):
    return Segment(text=text, start_line=1, end_line=1, kind="text", index=index)


def test_shingles_basic():
    result = shingles("the quick brown fox jumps over the lazy dog", size=3)
    assert ("the", "quick", "brown") in result
    assert ("jumps", "over", "the") in result
    assert len(result) == 7  # 9 words, size 3 -> 7 windows


def test_shingles_short_text_yields_one_shingle():
    result = shingles("one two", size=5)
    assert result == {("one", "two")}


def test_shingles_empty_text():
    assert shingles("", size=5) == set()
    assert shingles("   ...   ", size=5) == set()


def test_shingles_case_insensitive():
    assert shingles("Hello World", size=2) == shingles("hello world", size=2)


def test_shingles_rejects_nonpositive_size():
    try:
        shingles("text", size=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_jaccard_identical_sets():
    a = shingles("a b c d e f", size=3)
    assert jaccard(a, a) == 1.0


def test_jaccard_disjoint_sets():
    a = shingles("apple banana cherry", size=3)
    b = shingles("dog elephant fox", size=3)
    assert jaccard(a, b) == 0.0


def test_jaccard_both_empty():
    assert jaccard(set(), set()) == 1.0


def test_jaccard_one_empty():
    a = shingles("some text here", size=2)
    assert jaccard(a, set()) == 0.0
    assert jaccard(set(), a) == 0.0


def test_jaccard_partial_overlap():
    a = {("a", "b"), ("b", "c"), ("c", "d")}
    b = {("b", "c"), ("c", "d"), ("d", "e")}
    # intersection 2, union 4
    assert jaccard(a, b) == 0.5


def test_find_near_duplicates_flags_reworded_repeat():
    segments = [
        make_segment("Reading file config.yaml at 10:00:01, found 42 lines."),
        make_segment("Reading file config.yaml at 10:00:02, found 42 lines."),
        make_segment("Compiling the project with optimizations enabled now."),
    ]
    pairs = find_near_duplicates(segments, threshold=0.7, shingle_size=1)
    assert pairs == [(0, 1)]


def test_find_near_duplicates_empty_segments_never_match():
    segments = [make_segment(""), make_segment("")]
    assert find_near_duplicates(segments, threshold=0.5) == []


def test_dedupe_segments_keeps_first_occurrence():
    segments = [
        make_segment("Traceback (most recent call last): ValueError: bad input"),
        make_segment("Unrelated paragraph about something else entirely."),
        make_segment("Traceback (most recent call last): ValueError: bad input"),
    ]
    kept = dedupe_segments(segments, threshold=0.9, shingle_size=3)
    assert kept == [segments[0], segments[1]]


def test_dedupe_segments_preserves_order_and_originals():
    segments = [make_segment("alpha beta gamma", index=0), make_segment("delta epsilon zeta", index=1)]
    kept = dedupe_segments(segments)
    assert kept == segments


def test_dedupe_segments_high_threshold_keeps_similar_but_distinct():
    segments = [
        make_segment("The build failed on step one due to a missing dependency."),
        make_segment("The build failed on step two due to a missing dependency."),
    ]
    kept = dedupe_segments(segments, threshold=0.99, shingle_size=3)
    assert len(kept) == 2
