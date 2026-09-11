from ctx_squeeze.compactor import STRATEGIES, squeeze
from ctx_squeeze.segments import split_segments


def test_strategies_registry_has_the_documented_stage_names():
    assert set(STRATEGIES) == {"dedupe", "score", "head-tail"}


def test_squeeze_returns_original_when_under_budget():
    text = "alpha\n\nbeta\n\ngamma"
    result = squeeze(text, budget=10_000)
    assert result.text == text
    assert result.notes == []
    assert result.segments_in == result.segments_out == 3
    assert result.original_tokens == result.final_tokens


def test_squeeze_empty_text_returns_empty_result():
    result = squeeze("", budget=10)
    assert result.text == ""
    assert result.segments_in == 0
    assert result.segments_out == 0
    assert result.notes == []


def test_squeeze_rejects_unknown_strategy():
    try:
        squeeze("alpha\n\nbeta", budget=1, strategy="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_squeeze_rejects_blank_strategy():
    try:
        squeeze("alpha\n\nbeta", budget=1, strategy=" , ,")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_squeeze_score_strategy_respects_budget():
    text = "\n\n".join(
        [
            "kubernetes reconciler loop retries backoff jitter",
            "unrelated filler paragraph about something else entirely",
            "kubernetes reconciler loop backoff exponential",
        ]
    )
    budget = 8
    result = squeeze(text, budget, strategy="score")
    assert result.final_tokens <= budget
    assert result.segments_out < result.segments_in


def test_squeeze_head_tail_keeps_first_and_last_drops_middle():
    text = "one\n\ntwo\n\nthree"
    result = squeeze(text, budget=2, strategy="head-tail", marker=False)
    assert result.text == "one\n\nthree"
    assert result.segments_out == 2


def test_squeeze_dedupe_stage_drops_duplicate_and_notes():
    para1 = "Reading file config.yaml at 10:00:01, found 42 lines."
    para2 = "Reading file config.yaml at 10:00:02, found 42 lines."
    para3 = "Compiling the project with optimizations enabled now."
    text = "\n\n".join([para1, para2, para3])
    segments = split_segments(text)
    budget = segments[0].tokens + segments[2].tokens + 10  # room for the elision marker too

    result = squeeze(text, budget, strategy="dedupe", shingle_size=1, jaccard_threshold=0.7)

    assert "dedupe dropped 1 near-duplicate segment(s)" in result.notes
    assert para2 not in result.text
    assert para1 in result.text
    assert para3 in result.text
    assert result.final_tokens <= budget


def test_squeeze_marker_false_omits_elision_markers():
    para1 = "Reading file config.yaml at 10:00:01, found 42 lines."
    para2 = "Reading file config.yaml at 10:00:02, found 42 lines."
    para3 = "Compiling the project with optimizations enabled now."
    text = "\n\n".join([para1, para2, para3])
    segments = split_segments(text)
    budget = segments[0].tokens + segments[2].tokens

    result = squeeze(text, budget, strategy="dedupe", shingle_size=1, jaccard_threshold=0.7, marker=False)

    assert "[" not in result.text
    assert result.text == para1 + "\n\n" + para3


def test_squeeze_dedupe_then_score_chaining():
    para_a = "Reading file config.yaml at 10:00:01, found 42 lines."
    para_b = "Reading file config.yaml at 10:00:02, found 42 lines."
    para_c = "kubernetes reconciler loop backoff jitter retries"
    para_d = "the a an of and it is on at"
    text = "\n\n".join([para_a, para_b, para_c, para_d])

    result = squeeze(text, budget=27, strategy="dedupe,score", shingle_size=1, jaccard_threshold=0.7)

    assert "dedupe dropped 1 near-duplicate segment(s)" in result.notes
    assert para_b not in result.text
    assert para_d not in result.text
    assert result.final_tokens <= 27


def test_squeeze_trims_kept_segments_when_markers_exceed_budget():
    para1 = "Reading file config.yaml at 10:00:01, found 42 lines."
    para2 = "Reading file config.yaml at 10:00:02, found 42 lines."
    para3 = "Compiling the project with optimizations enabled now."
    text = "\n\n".join([para1, para2, para3])
    segments = split_segments(text)
    budget = segments[0].tokens + segments[2].tokens  # no room left for the marker

    result = squeeze(text, budget, strategy="dedupe", shingle_size=1, jaccard_threshold=0.7)

    assert any("additional segment(s) to keep elision markers within budget" in note for note in result.notes)
    assert result.final_tokens <= budget
    assert result.segments_out == 1
