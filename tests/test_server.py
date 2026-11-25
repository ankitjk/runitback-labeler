"""The server's tools, run against a tiny made-up video and game list in a temporary folder."""

import json

import cv2
import numpy as np
import pytest

from labeler import config, server

VIDEO_ID = "TESTvideo01"


@pytest.fixture
def world(tmp_path, monkeypatch):
    videos, truth, output = tmp_path / "videos", tmp_path / "truth", tmp_path / "output"
    for folder in (videos, truth):
        folder.mkdir()
    monkeypatch.setattr(config, "VIDEO_CACHE_DIR", videos)
    monkeypatch.setattr(config, "GROUND_TRUTH_DIR", truth)
    monkeypatch.setattr(config, "OUTPUT_DIR", output)

    writer = cv2.VideoWriter(str(videos / f"{VIDEO_ID}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 180))
    for i in range(50):  # 5 seconds
        writer.write(np.full((180, 320, 3), i * 5, dtype=np.uint8))
    writer.release()

    (truth / f"{VIDEO_ID}.json").write_text(json.dumps({
        "video_id": VIDEO_ID,
        "video_path": f".video_cache/{VIDEO_ID}.mp4",
        "games": [{"game_number": 1, "stage": "battlefield", "start_sec": 1.0, "end_sec": 4.0,
                   "labels": [{"start_sec": 1.0, "end_sec": 4.0, "class": "both_grounded"}]}],
    }))
    return output


def seg(start, end, cls="both_grounded", source="model", **extra):
    return {"start_sec": start, "end_sec": end, "class": cls, "source": source, **extra}


def test_game_bounds_never_returns_labels(world):
    bounds = server.game_bounds(VIDEO_ID, 1)
    assert bounds == {"video_path": f".video_cache/{VIDEO_ID}.mp4", "stage": "battlefield", "start_sec": 1.0, "end_sec": 4.0}


def test_unknown_game_and_unknown_video_are_refused(world):
    with pytest.raises(ValueError, match="has no game"):
        server.game_bounds(VIDEO_ID, 2)
    with pytest.raises(ValueError, match="no video"):
        server.get_frame("NOSUCHvideo", 1.0)
    with pytest.raises(ValueError, match="no video"):
        server.save_draft_labels("NOSUCHvideo", 1, [seg(1, 2)])
    with pytest.raises(ValueError, match="not a valid id"):
        server.predict_segments("../../secret", 1)


def test_get_frame_returns_a_jpeg_and_respects_max_width(world):
    image = server.get_frame(VIDEO_ID, 2.0, max_width=160)
    assert image.data[:2] == b"\xff\xd8"
    decoded = cv2.imdecode(np.frombuffer(image.data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (90, 160)


@pytest.mark.parametrize("t", [-1.0, 5.0, 999.0])
def test_get_frame_refuses_times_outside_the_video(world, t):
    with pytest.raises(ValueError, match="outside the video"):
        server.get_frame(VIDEO_ID, t)


def test_good_save_writes_the_label_format(world):
    result = server.save_draft_labels(VIDEO_ID, 1, [seg(1.0, 2.5), seg(2.5, 4.0, "one_airborne", "llm")])
    assert result["saved"] is True
    draft = json.loads((world / f"{VIDEO_ID}_g1_draft.json").read_text())
    assert draft["video_id"] == VIDEO_ID
    game = draft["games"][0]
    assert (game["game_number"], game["stage"], game["start_sec"], game["end_sec"]) == (1, "battlefield", 1.0, 4.0)
    assert [label["class"] for label in game["labels"]] == ["both_grounded", "one_airborne"]


def test_refused_save_writes_nothing(world):
    result = server.save_draft_labels(VIDEO_ID, 1, [seg(1.0, 2.0, "finishing_blow")])
    assert result["saved"] is False and result["errors"]
    assert not world.exists() or list(world.iterdir()) == []


def test_refused_save_leaves_the_earlier_draft_alone(world):
    server.save_draft_labels(VIDEO_ID, 1, [seg(1.0, 4.0)])
    before = (world / f"{VIDEO_ID}_g1_draft.json").read_text()
    assert server.save_draft_labels(VIDEO_ID, 1, [seg(0.0, 9.0)])["saved"] is False
    assert (world / f"{VIDEO_ID}_g1_draft.json").read_text() == before
    assert [p.name for p in world.iterdir()] == [f"{VIDEO_ID}_g1_draft.json"]


def test_a_later_save_cannot_change_a_human_segment(world):
    human = seg(1.0, 2.0, "one_offstage", "human")
    assert server.save_draft_labels(VIDEO_ID, 1, [human, seg(2.0, 4.0)])["saved"] is True
    changed = server.save_draft_labels(VIDEO_ID, 1, [seg(1.0, 2.0, "both_offstage", "model"), seg(2.0, 4.0)])
    assert changed["saved"] is False and "human-labelled segment" in changed["errors"][0]
    kept = server.save_draft_labels(VIDEO_ID, 1, [human, seg(2.0, 4.0, "one_airborne", "llm")])
    assert kept["saved"] is True


def test_merge_joins_neighbouring_frames():
    times = server._sample_times(10.0, 12.0, 2)
    assert times == [10.25, 10.75, 11.25, 11.75]
    predictions = [("both_grounded", 0.9), ("both_grounded", 0.5), ("one_airborne", 0.6), ("both_grounded", 0.7)]
    segments = server._merge(times, predictions, 10.0, 12.0, 2)
    assert [(s["class"], s["start_sec"], s["end_sec"], s["n_frames"]) for s in segments] == [
        ("both_grounded", 10.0, 11.0, 2), ("one_airborne", 11.0, 11.5, 1), ("both_grounded", 11.5, 12.0, 1)]
    assert segments[0]["mean_confidence"] == 0.7 and segments[0]["min_confidence"] == 0.5


@pytest.mark.parametrize("bad", [0, 11, 2.5, "4", True])
def test_predict_refuses_a_bad_sample_rate(world, bad):
    with pytest.raises(ValueError, match="sample_fps"):
        server.predict_segments(VIDEO_ID, 1, sample_fps=bad)
