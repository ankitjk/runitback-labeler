"""The workflow: a LangGraph graph that walks through a game's segments one at a time.

    predict -> next_segment -+-> accept -+
                    ^        +-> defer --+
                    +--------------------+
                             +-> save_draft -> END
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from labeler import config, routing

# Fields copied from the workflow's segment into the draft, when present.
DRAFT_EXTRA_FIELDS = ("needs_review", "model_label", "mean_confidence", "min_confidence")


class State(TypedDict, total=False):
    video_id: str
    game_number: int
    segments: list[dict]
    cursor: int  # index of the segment being handled
    questions_asked: int
    save_result: dict


def _with_current_updated(state: State, **changes) -> dict:
    """Return a state update that changes only the current segment."""
    segments = list(state["segments"])
    segments[state["cursor"]] = {**segments[state["cursor"]], **changes}
    return {"segments": segments}


def build_graph(client, threshold: float | None = None, checkpointer=None):
    """Build the workflow. `client` is anything with the three MCP tool methods."""
    threshold = config.CONFIDENCE_THRESHOLD if threshold is None else threshold

    async def predict(state: State) -> dict:
        result = await client.predict_segments(state["video_id"], state["game_number"])
        segments = [
            {
                "start_sec": s["start_sec"],
                "end_sec": s["end_sec"],
                "model_label": s["class"],
                "mean_confidence": s["mean_confidence"],
                "min_confidence": s["min_confidence"],
            }
            for s in result["segments"]
        ]
        return {"segments": segments, "cursor": -1, "questions_asked": 0}

    def next_segment(state: State) -> dict:
        return {"cursor": state["cursor"] + 1}

    def route(state: State) -> str:
        if state["cursor"] >= len(state["segments"]):
            return "save_draft"
        segment = state["segments"][state["cursor"]]
        decision = routing.after_predict(segment, threshold, config.NEVER_AUTO_ACCEPT)
        # Until the LLM step exists, a segment that needs a second opinion is deferred.
        return "accept" if decision == "accept" else "defer"

    def accept(state: State) -> dict:
        segment = state["segments"][state["cursor"]]
        return _with_current_updated(state, final_label=segment["model_label"], source="model")

    def defer(state: State) -> dict:
        segment = state["segments"][state["cursor"]]
        return _with_current_updated(state, final_label=segment["model_label"], source="model", needs_review=True)

    async def save_draft(state: State) -> dict:
        draft_segments = []
        for segment in state["segments"]:
            entry = {
                "start_sec": segment["start_sec"],
                "end_sec": segment["end_sec"],
                "class": segment["final_label"],
                "source": segment["source"],
            }
            entry.update({field: segment[field] for field in DRAFT_EXTRA_FIELDS if field in segment})
            draft_segments.append(entry)
        result = await client.save_draft_labels(state["video_id"], state["game_number"], draft_segments)
        return {"save_result": result}

    graph = StateGraph(State)
    graph.add_node("predict", predict)
    graph.add_node("next_segment", next_segment)
    graph.add_node("accept", accept)
    graph.add_node("defer", defer)
    graph.add_node("save_draft", save_draft)

    graph.add_edge(START, "predict")
    graph.add_edge("predict", "next_segment")
    graph.add_conditional_edges("next_segment", route, ["accept", "defer", "save_draft"])
    graph.add_edge("accept", "next_segment")
    graph.add_edge("defer", "next_segment")
    graph.add_edge("save_draft", END)
    return graph.compile(checkpointer=checkpointer)
