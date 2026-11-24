"""The workflow, run with a stand-in for the MCP server."""

import asyncio

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
