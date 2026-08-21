from ctx_squeeze.scoring import score_segments, select_by_score
from ctx_squeeze.segments import Segment


def make_segment(text, index=0):
    return Segment(text=text, start_line=1, end_line=1, kind="text", index=index)


def test_score_segments_empty_list():
    assert score_segments([]) == []


def test_score_segments_returns_one_score_per_segment():
    segments = [make_segment("alpha beta"), make_segment("gamma delta epsilon")]
    scores = score_segments(segments)
    assert len(scores) == 2


def test_score_segments_empty_segment_scores_zero():
    segments = [make_segment(""), make_segment("real content here")]
    scores = score_segments(segments)
    assert scores[0] == 0.0


def test_score_segments_stopwords_only_scores_zero():
    segments = [make_segment("the a an of and"), make_segment("kubernetes reconciler loop")]
    scores = score_segments(segments)
    assert scores[0] == 0.0


def test_score_segments_rare_word_scores_higher_than_common_word():
    # "runner" appears in every segment (no signal); the other two mix in words
    # that appear only once, which should outweigh a segment of "runner" alone.
    segments = [
        make_segment("runner runner runner"),
        make_segment("runner postmortem forensics"),
        make_segment("runner cache restored"),
    ]
    scores = score_segments(segments)
    assert scores[1] > scores[0]
    assert scores[2] > scores[0]


def test_score_segments_identical_segments_score_equally():
    segments = [make_segment("build failed on step one"), make_segment("build failed on step one")]
    scores = score_segments(segments)
    assert scores[0] == scores[1]


def test_select_by_score_empty_budget_returns_nothing():
    segments = [make_segment("some content"), make_segment("more content")]
    assert select_by_score(segments, budget=0) == []


def test_select_by_score_empty_segments_returns_nothing():
    assert select_by_score([], budget=100) == []


def test_select_by_score_respects_budget():
    segments = [
        make_segment("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"),
        make_segment("nu xi omicron pi rho sigma tau upsilon phi chi psi omega"),
        make_segment("short"),
    ]
    budget = 6
    selected = select_by_score(segments, budget)
    assert sum(segment.tokens for segment in selected) <= budget


def test_select_by_score_preserves_document_order():
    segments = [
        make_segment("kubernetes reconciler loop retries", index=0),
        make_segment("unrelated filler paragraph text words", index=1),
        make_segment("kubernetes reconciler loop backoff", index=2),
    ]
    selected = select_by_score(segments, budget=1000)
    assert selected == segments


def test_select_by_score_skips_expensive_low_scoring_segment_for_cheaper_ones():
    # A long segment with mostly repeated/common words versus two small,
    # keyword-dense segments that together fit the same budget.
    filler = " ".join(["the"] * 50)
    segments = [
        make_segment(filler, index=0),
        make_segment("postmortem", index=1),
        make_segment("remediation", index=2),
    ]
    budget = min(segments[1].tokens, segments[2].tokens) + max(segments[1].tokens, segments[2].tokens)
    selected = select_by_score(segments, budget)
    assert segments[0] not in selected
    assert segments[1] in selected
    assert segments[2] in selected


def test_select_by_score_ties_break_by_original_position():
    segments = [make_segment("alpha beta"), make_segment("gamma delta")]
    budget = segments[0].tokens
    selected = select_by_score(segments, budget)
    assert selected == [segments[0]]
