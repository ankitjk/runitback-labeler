import pytest

from labeler import evaluate as ev

NEVER_FREE = dict(model_label="both_grounded")


def truth(*rows):
    return [{"start_sec": a, "end_sec": b, "class": c} for a, b, c in rows]


def model(*rows):
    return [{"start_sec": a, "end_sec": b, "model_label": c, "min_confidence": conf} for a, b, c, conf in rows]


def test_points_cover_every_tenth_of_a_second_that_a_human_labelled():
    t = truth((10.0, 10.5, "both_grounded"), (10.6, 11.0, "one_airborne"))  # 0.1 s gap at 10.5
    m = model((10.0, 11.0, "both_grounded", 0.9))
    points = ev.build_points(t, m, 10.0, 11.0)
    assert len(points) == 9
    assert [p["t"] for p in points][:2] == [10.05, 10.15]
    assert all(p["segment"] == 0 for p in points)


def test_accuracy_is_the_share_of_time_that_matches():
    t = truth((0.0, 1.0, "both_grounded"), (1.0, 2.0, "one_airborne"))
    m = model((0.0, 1.5, "both_grounded", 0.9), (1.5, 2.0, "one_airborne", 0.9))
    points = ev.build_points(t, m, 0.0, 2.0)
    assert ev.accuracy(points) == pytest.approx(0.75)


def test_per_class_recall_and_precision():
    t = truth((0.0, 1.0, "both_grounded"), (1.0, 2.0, "one_airborne"))
    m = model((0.0, 1.5, "both_grounded", 0.9), (1.5, 2.0, "one_airborne", 0.9))
    rows = {r["class"]: r for r in ev.per_class(ev.build_points(t, m, 0.0, 2.0))}
    assert rows["both_grounded"]["recall"] == pytest.approx(1.0)
    assert rows["both_grounded"]["precision"] == pytest.approx(10 / 15)
    assert rows["one_airborne"]["recall"] == pytest.approx(0.5)
    assert "respawning" not in rows


def test_segmentation_ceiling_uses_the_best_label_per_model_segment():
    # One model segment spans two human classes 70/30: the best it can score is 70%.
    t = truth((0.0, 0.7, "both_grounded"), (0.7, 1.0, "one_airborne"))
    m = model((0.0, 1.0, "respawning", 0.9))
    points = ev.build_points(t, m, 0.0, 1.0)
    assert ev.accuracy(points) == 0.0
    assert ev.segmentation_ceiling(points) == pytest.approx(0.7)


def test_sweep_counts_coverage_and_error():
    t = truth((0.0, 3.0, "both_grounded"))
    m = model((0.0, 1.0, "both_grounded", 0.9), (1.0, 2.0, "one_airborne", 0.5), (2.0, 3.0, "respawning", 0.95))
    points = ev.build_points(t, m, 0.0, 3.0)
    low, high = ev.sweep(points, m, [0.4, 0.8])
    # respawning is never accepted on the model's word, however confident
    assert low["accepted_segments"] == 2 and low["second_opinion_segments"] == 1
    assert low["coverage"] == pytest.approx(2 / 3) and low["auto_accept_error"] == pytest.approx(0.5)
    assert high["accepted_segments"] == 1 and high["auto_accept_error"] == 0.0


def test_sweep_with_nothing_accepted_has_no_error_rate():
    t = truth((0.0, 1.0, "both_grounded"))
    m = model((0.0, 1.0, "both_grounded", 0.2))
    row = ev.sweep(ev.build_points(t, m, 0.0, 1.0), m, [0.9])[0]
    assert row["coverage"] == 0.0 and row["auto_accept_error"] is None


def test_training_games_are_refused():
    with pytest.raises(ValueError, match="not an unseen game"):
        ev.load_truth("XUkRx6CpL04", 1)
    with pytest.raises(ValueError, match="not an unseen game"):
        ev.load_truth("XUkRx6CpL04", 2)
