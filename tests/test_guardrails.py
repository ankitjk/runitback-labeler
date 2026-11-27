import pytest

from labeler.guardrails import check_llm_names, check_video_id, draft_path, validate_segments

START, END, TOL = 100.0, 200.0, 0.05


def seg(start, end, cls="both_grounded", source="model", **extra):
    return {"start_sec": start, "end_sec": end, "class": cls, "source": source, **extra}


def check(segments, existing=None):
    return validate_segments(segments, START, END, TOL, existing)


def test_valid_draft_passes():
    segments = [
        seg(100.0, 110.0, "non_gameplay"),
        seg(110.0, 110.5, "one_airborne", "human", model_label="both_airborne", min_confidence=0.4),
        seg(110.5, 200.0, "respawning", "llm", needs_review=True),
    ]
    assert check(segments) == []


def test_gaps_between_segments_are_allowed():
    assert check([seg(100, 110), seg(120, 130)]) == []


def test_empty_or_wrong_type_is_refused():
    assert check([]) != []
    assert check("not a list") != []
    assert "must be an object" in check(["oops"])[0]


def test_unknown_class_is_refused():
    errors = check([seg(100, 110, "finishing_blow")])
    assert errors == ["segment 0 (100.00-110.00): class 'finishing_blow' is not one of the 11 classes"]


def test_missing_or_bad_source_is_refused():
    no_source = {"start_sec": 100, "end_sec": 110, "class": "both_grounded"}
    assert "missing 'source'" in check([no_source])[0]
    assert "source 'robot'" in check([seg(100, 110, source="robot")])[0]


def test_unknown_field_is_refused():
    assert "unknown field 'note'" in check([seg(100, 110, note="hi")])[0]


def test_end_must_be_after_start():
    assert "end_sec must be after start_sec" in check([seg(110, 110)])[0]
    assert "end_sec must be after start_sec" in check([seg(110, 105)])[0]


def test_times_must_be_numbers():
    assert "must be numbers" in check([seg("100", 110)])[0]
    assert "must be numbers" in check([seg(100, float("nan"))])[0]
    assert "must be numbers" in check([seg(True, 110)])[0]


def test_bounds_with_tolerance():
    assert check([seg(99.96, 200.04)]) == []
    assert "outside the game's bounds" in check([seg(99.9, 110)])[0]
    assert "outside the game's bounds" in check([seg(190, 200.1)])[0]


def test_overlap_is_refused():
    errors = check([seg(100, 110), seg(109.5, 120)])
    assert errors == ["segment 1 (109.50-120.00): overlaps the segment before it, which ends at 110.00"]


def test_out_of_order_is_refused():
    assert "out of order" in check([seg(120, 130), seg(100, 110)])[0]


def test_every_broken_rule_is_reported():
    errors = check([seg(100, 110, "nope", "robot"), seg(105, 300)])
    assert len(errors) == 4


def test_needs_review_must_be_boolean():
    assert "needs_review" in check([seg(100, 110, needs_review="yes")])[0]


def test_human_segments_cannot_be_changed_or_dropped():
    existing = [seg(100, 110, "one_offstage", "human"), seg(110, 120, "both_grounded", "model")]

    unchanged = [seg(100, 110, "one_offstage", "human"), seg(110, 120, "one_airborne", "llm")]
    assert check(unchanged, existing) == []

    relabelled = [seg(100, 110, "both_offstage", "human"), seg(110, 120)]
    assert "human-labelled segment" in check(relabelled, existing)[0]

    taken_over = [seg(100, 110, "one_offstage", "model"), seg(110, 120)]
    assert "human-labelled segment" in check(taken_over, existing)[0]

    dropped = [seg(110, 120)]
    assert "human-labelled segment" in check(dropped, existing)[0]


def test_human_segment_extra_fields_cannot_be_changed():
    existing = [seg(100, 110, "one_offstage", "human", model_label="one_on_ledge", llm_reason="below the edge")]
    assert check([dict(existing[0])], existing) == []
    reworded = [dict(existing[0], llm_reason="something else")]
    assert "human-labelled segment" in check(reworded, existing)[0]
    flagged = [dict(existing[0], needs_review=True)]
    assert "human-labelled segment" in check(flagged, existing)[0]


@pytest.mark.parametrize("bad", ["../../etc/passwd", "short", "XUkRx6CpL04/..", "", None, "XUkRx6CpL04.mp4"])
def test_video_id_cannot_be_a_path(bad):
    with pytest.raises(ValueError):
        check_video_id(bad)


def test_draft_path_stays_inside_output(tmp_path):
    path = draft_path(tmp_path, "XUkRx6CpL04", 3)
    assert path == tmp_path.resolve() / "XUkRx6CpL04_g3_draft.json"
    for bad_game in (0, -1, "3", 3.0, True):
        with pytest.raises(ValueError):
            draft_path(tmp_path, "XUkRx6CpL04", bad_game)
    with pytest.raises(ValueError):
        draft_path(tmp_path, "../escape", 3)


def test_draft_path_refuses_a_link_that_leaves_output(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    outside = tmp_path / "elsewhere.json"
    outside.write_text("{}")
    (output / "XUkRx6CpL04_g3_draft.json").symlink_to(outside)
    with pytest.raises(ValueError):
        draft_path(output, "XUkRx6CpL04", 3)


def test_llm_names_must_be_plain_names():
    assert check_llm_names("", "") == []  # no LLM was used
    assert check_llm_names("gemini-4-hydrogen", "v1_abcd1234") == []
    assert len(check_llm_names("../../x", "v1_abcd1234")) == 1
    assert len(check_llm_names("some-model", "a prompt id")) == 1
    assert len(check_llm_names(None, 7)) == 2
