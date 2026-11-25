"""The workflow, run with a stand-in for the MCP server."""

import asyncio

from labeler import config
from labeler.graph import build_graph


class FakeClient:
    def __init__(self, segments, save_result=None):
        self.segments = segments
        self.save_result = save_result or {"saved": True, "path": "output/fake.json"}
        self.saved = None

    async def predict_segments(self, video_id, game_number):
        return {"segments": self.segments}

    async def save_draft_labels(self, video_id, game_number, segments):
        self.saved = segments
        return self.save_result


def model_segment(start, cls, confidence):
    return {"class": cls, "start_sec": start, "end_sec": start + 1, "mean_confidence": confidence,
            "min_confidence": confidence, "n_frames": 2}


def run(client, threshold=0.6):
    graph = build_graph(client, threshold=threshold)
    return asyncio.run(graph.ainvoke({"video_id": "XUkRx6CpL04", "game_number": 3}, config={"recursion_limit": 100}))


def test_each_segment_takes_the_right_route():
    client = FakeClient([
        model_segment(0, "non_gameplay", 0.9),   # confident -> accepted
        model_segment(1, "one_airborne", 0.3),   # unsure -> deferred
        model_segment(2, "respawning", 0.95),    # over-predicted class -> deferred
    ])
    state = run(client)

    assert [s["final_label"] for s in state["segments"]] == ["non_gameplay", "one_airborne", "respawning"]
    assert [s.get("needs_review", False) for s in state["segments"]] == [False, True, True]
    assert state["cursor"] == 3
    assert state["save_result"]["saved"] is True


def test_the_draft_sent_to_the_server_has_the_label_format():
    client = FakeClient([model_segment(0, "non_gameplay", 0.9), model_segment(1, "one_airborne", 0.3)])
    run(client)

    assert client.saved[0] == {"start_sec": 0, "end_sec": 1, "class": "non_gameplay", "source": "model",
                               "model_label": "non_gameplay", "mean_confidence": 0.9, "min_confidence": 0.9}
    assert client.saved[1]["needs_review"] is True
    assert all("final_label" not in s and "n_frames" not in s for s in client.saved)


def test_a_refused_save_is_reported_in_the_state():
    client = FakeClient([model_segment(0, "non_gameplay", 0.9)], save_result={"saved": False, "errors": ["nope"]})
    assert run(client)["save_result"] == {"saved": False, "errors": ["nope"]}


def test_many_segments_do_not_hit_the_step_limit():
    client = FakeClient([model_segment(i, "non_gameplay", 0.9) for i in range(300)])
    graph = build_graph(client, threshold=0.6)
    state = asyncio.run(graph.ainvoke({"video_id": "XUkRx6CpL04", "game_number": 3}, config={"recursion_limit": 10_000}))
    assert len(client.saved) == 300 and state["cursor"] == 300


class FakeLLM:
    """Answers from a table keyed by frame time, and records which frames it was asked about."""

    def __init__(self, answers):
        self.answers = answers
        self.asked = []

    async def label_frame(self, video_id, t_sec, get_jpeg):
        self.asked.append(t_sec)
        assert await get_jpeg() == b"jpeg"
        label = self.answers[t_sec]
        return {"llm_label": label, "llm_zoom": False, "llm_reason": "because" if label else "",
                "llm_error": None if label else "bad answer"}


class FakeClientWithFrames(FakeClient):
    async def get_frame(self, video_id, t_sec):
        return b"jpeg"


def test_with_an_llm_segments_take_all_three_routes(monkeypatch):
    client = FakeClientWithFrames([
        model_segment(0, "non_gameplay", 0.9),   # confident -> accepted, LLM not asked
        model_segment(1, "respawning", 0.9),     # LLM agrees -> accepted
        model_segment(2, "one_airborne", 0.3),   # LLM disagrees -> deferred
        model_segment(3, "one_airborne", 0.3),   # LLM's answer failed the checks -> deferred
    ])
    monkeypatch.setattr(config, "DEFERRED_LABEL", "model")
    llm = FakeLLM({1.5: "respawning", 2.5: "both_airborne", 3.5: None})
    graph = build_graph(client, llm=llm, threshold=0.6)
    state = asyncio.run(graph.ainvoke({"video_id": "XUkRx6CpL04", "game_number": 3}, config={"recursion_limit": 100}))

    assert llm.asked == [1.5, 2.5, 3.5]  # the middle of each unsure segment, and never the confident one
    assert [s.get("needs_review", False) for s in state["segments"]] == [False, False, True, True]
    assert [s["final_label"] for s in state["segments"]] == ["non_gameplay", "respawning", "one_airborne", "one_airborne"]
    assert all(s["source"] == "model" for s in state["segments"])

    assert "llm_label" not in client.saved[0]
    assert client.saved[1]["llm_label"] == "respawning" and client.saved[1]["llm_reason"] == "because"
    assert client.saved[2]["llm_label"] == "both_airborne"
    assert "llm_label" not in client.saved[3] and "llm_reason" not in client.saved[3]
    assert all("llm_error" not in s and "llm_zoom" not in s for s in client.saved)


def test_a_deferred_segment_can_carry_the_llm_label_and_says_so(monkeypatch):
    monkeypatch.setattr(config, "DEFERRED_LABEL", "llm")
    client = FakeClientWithFrames([model_segment(0, "one_airborne", 0.3), model_segment(1, "one_airborne", 0.3)])
    llm = FakeLLM({0.5: "both_airborne", 1.5: None})
    graph = build_graph(client, llm=llm, threshold=0.6)
    asyncio.run(graph.ainvoke({"video_id": "XUkRx6CpL04", "game_number": 3}, config={"recursion_limit": 100}))

    # The LLM gave a class: the draft carries it, marked as the LLM's and as needing review.
    assert (client.saved[0]["class"], client.saved[0]["source"], client.saved[0]["needs_review"]) == ("both_airborne", "llm", True)
    # The LLM gave no usable class: the model's label stays.
    assert (client.saved[1]["class"], client.saved[1]["source"], client.saved[1]["needs_review"]) == ("one_airborne", "model", True)

