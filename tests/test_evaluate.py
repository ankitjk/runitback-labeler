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


def with_llm(segments, *labels):
    return [{**s, "llm_label": label, "llm_zoom": False, "llm_error": None if label else "bad"}
            for s, label in zip(segments, labels)]


def llm_case():
    # Human: grounded for 2 s, then airborne for 2 s. Four 1 s model segments, all unsure (0.3).
    t = truth((0.0, 2.0, "both_grounded"), (2.0, 4.0, "one_airborne"))
    m = model((0.0, 1.0, "both_grounded", 0.3), (1.0, 2.0, "respawning", 0.3),
              (2.0, 3.0, "respawning", 0.3), (3.0, 4.0, "both_grounded", 0.3))
    #                agree+right      disagree, LLM right  agree+wrong   no usable answer
    m = with_llm(m, "both_grounded", "both_grounded", "respawning", None)
    return ev.build_points(t, m, 0.0, 4.0), m


def test_llm_accuracy_counts_a_missing_answer_as_wrong():
    points, m = llm_case()
    assert ev.llm_accuracy(points, m) == pytest.approx(0.5)


def test_agreement_reports_how_often_both_are_wrong_together():
    points, m = llm_case()
    everything, on_ledge, respawning = ev.agreement(points, m)
    assert everything == {"where": "all classes", "sec": 2.0, "wrong_sec": 1.0, "error": pytest.approx(0.5)}
    assert respawning == {"where": "respawning", "sec": 1.0, "wrong_sec": 1.0, "error": 1.0}
    assert on_ledge["sec"] == 0.0 and on_ledge["error"] is None


def test_disagreement_counts_who_matched_the_human():
    points, m = llm_case()
    assert ev.disagreement(points, m) == {"sec": 1.0, "model_right_sec": 0.0, "llm_right_sec": 1.0, "neither_sec": 0.0}


def test_sweep_with_llm_accepts_agreements_and_leaves_the_rest_for_the_human():
    points, m = llm_case()
    (row,) = ev.sweep_with_llm(points, m, [0.4])
    assert row["accepted_model_alone"] == 0 and row["accepted_llm_agrees"] == 2 and row["left_for_human"] == 2
    assert row["coverage"] == pytest.approx(0.5) and row["auto_accept_error"] == pytest.approx(0.5)
    # With no threshold the confident-enough grounded segments are accepted on the model alone.
    (row,) = ev.sweep_with_llm(points, m, [0.0])
    assert row["accepted_model_alone"] == 2 and row["accepted_llm_agrees"] == 1 and row["left_for_human"] == 1


def test_the_llm_report_names_the_model_and_the_prompt():
    points, m = llm_case()
    report = ev.llm_report(points, m, "some-model", "v1_abcd1234")
    assert report["model_id"] == "some-model" and report["prompt_id"] == "v1_abcd1234"
    assert "some-model, prompt v1_abcd1234" in "\n".join(ev._llm_markdown(report, 0.5, 0.8))


def test_a_zoom_is_neither_an_agreement_nor_a_disagreement():
    points, m = llm_case()
    m = [{**s, "llm_zoom": True} for s in m]
    assert ev.agreement(points, m)[0]["sec"] == 0.0
    assert ev.disagreement(points, m)["sec"] == 0.0


def model_folder(tmp_path, name, content=b"a model", run_record=None):
    folder = tmp_path / name
    folder.mkdir()
    (folder / "scene_classifier.onnx").write_bytes(content)
    if run_record is not None:
        (folder / "run.json").write_text(run_record)
    return folder


def test_unseen_games_come_from_the_models_run_record(tmp_path):
    model = model_folder(tmp_path, "some_run", run_record='{"split": {"test_games": ["VID_a:3", "VID_b:1"]}}')
    assert ev.unseen_games(model) == {("VID_a", 3), ("VID_b", 1)}


def test_a_model_listed_in_config_ignores_a_run_record_in_its_folder(tmp_path, monkeypatch):
    # A file dropped into the folder must not be able to open up a training game.
    model = model_folder(tmp_path, "listed", run_record='{"split": {"test_games": ["XUkRx6CpL04:1"]}}')
    listed = {ev.model_file_sha256(model): frozenset({("XUkRx6CpL04", 3)})}
    monkeypatch.setattr(ev.config, "UNSEEN_GAMES_BY_MODEL", listed)
    assert ev.unseen_games(model) == {("XUkRx6CpL04", 3)}


def test_a_folder_name_alone_earns_nothing(tmp_path):
    assert ev.unseen_games(model_folder(tmp_path, "heldout_g3")) == frozenset()


def test_an_unknown_model_has_no_unseen_games_so_nothing_is_scored(tmp_path, monkeypatch):
    model = model_folder(tmp_path, "mystery")
    assert ev.unseen_games(model) == frozenset()
    assert ev.unseen_games(tmp_path / "no_such_folder") == frozenset()
    monkeypatch.setattr(ev.config, "MODEL_DIR", model)
    monkeypatch.setattr(ev.config, "MODEL_NAME", "mystery")
    with pytest.raises(ValueError, match="not an unseen game"):
        ev.load_truth("XUkRx6CpL04", 3)
