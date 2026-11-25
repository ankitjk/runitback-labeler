"""The workflow: a LangGraph graph that walks through a game's segments one at a time.

    predict -> next_segment -+-> accept ----------------------+
                    ^        +-> llm_opinion -+-> accept -----+
                    |        |                +-> human ------+   (pauses the run)
                    |        |                +-> defer ------+
                    +--------|----------------------------------+
                             +-> save_draft -> END

A segment the model can be trusted with is accepted. Any other goes to the
first-opinion LLM. If it agrees with the model the segment is accepted. If
not, the run pauses and a person is asked. Once the question cap is reached,
such segments are deferred: written to the draft marked `needs_review`.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from labeler import config, routing
from labeler.classes import CLASSES

SKIP = "skip"

# Fields copied from the workflow's segment into the draft, when present.
DRAFT_EXTRA_FIELDS = ("needs_review", "model_label", "mean_confidence", "min_confidence", "llm_label", "llm_reason")


class State(TypedDict, total=False):
    video_id: str
    game_number: int
    segments: list[dict]
    cursor: int  # index of the segment being handled
    questions_asked: int
    save_result: dict


def frame_time(segment: dict) -> float:
    """The moment of a segment that the LLM is shown: its middle."""
    return round((segment["start_sec"] + segment["end_sec"]) / 2, 3)


def human_choice(text: str, segment: dict) -> str | None:
    """Turn what the person typed into a class, SKIP, or None if it is not a valid answer.

    "1" is the model's label, "2" the LLM's (when it gave one), "s" skips;
    anything else must be the exact name of one of the 11 classes.
    """
    text = str(text).strip().lower()
    if text == "1":
        return segment["model_label"]
    if text == "2":
        return segment.get("llm_label")  # None if the LLM gave no class
    if text in ("s", SKIP):
        return SKIP
    return text if text in CLASSES else None


def _with_current_updated(state: State, **changes) -> dict:
    """Return a state update that changes only the current segment."""
    segments = list(state["segments"])
    segments[state["cursor"]] = {**segments[state["cursor"]], **changes}
    return {"segments": segments}


def build_graph(client, llm=None, threshold: float | None = None, checkpointer=None, max_questions: int | None = None):
    """Build the workflow.

    `client` is anything with the MCP tool methods. `llm` is anything with
    `label_frame(video_id, t_sec, get_jpeg)`; without one, unsure segments are deferred.
    Asking a person needs a `checkpointer`: it holds the state while the run is paused.
    """
    threshold = config.CONFIDENCE_THRESHOLD if threshold is None else threshold
    max_questions = config.MAX_QUESTIONS if max_questions is None else max_questions
    if checkpointer is None:
        max_questions = 0  # a run that cannot pause cannot ask, so everything for the human is deferred

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
        if decision == "accept":
            return "accept"
        return "llm_opinion" if llm is not None else "defer"

    async def llm_opinion(state: State) -> dict:
        """Show the LLM the segment's middle frame. Its answer is checked in code inside label_frame."""
        segment = state["segments"][state["cursor"]]
        t_sec = frame_time(segment)
        opinion = await llm.label_frame(
            state["video_id"], t_sec, lambda: client.get_frame(state["video_id"], t_sec)
        )
        return _with_current_updated(state, **opinion)

    def route_llm(state: State) -> str:
        segment = state["segments"][state["cursor"]]
        decision = routing.after_llm(segment, state["questions_asked"], max_questions, config.USE_CONTEXT_AGENT)
        # The context agent is not built yet, so its segments are deferred.
        return "defer" if decision == "context" else decision

    def human(state: State) -> dict:
        """Pause the run and ask a person. Nothing may happen before interrupt():
        when the run resumes, this function starts again from its first line."""
        segment = state["segments"][state["cursor"]]
        answer = interrupt({
            "video_id": state["video_id"],
            "t_sec": frame_time(segment),
            "start_sec": segment["start_sec"],
            "end_sec": segment["end_sec"],
            "model_label": segment["model_label"],
            "min_confidence": segment["min_confidence"],
            "llm_label": segment.get("llm_label"),
            "llm_zoom": segment.get("llm_zoom", False),
            "llm_reason": segment.get("llm_reason", ""),
            "question_number": state["questions_asked"] + 1,
            "max_questions": max_questions,
        })
        choice = human_choice(answer, segment)
        if choice is None:
            raise ValueError(f"{answer!r} is not a valid answer for this segment")
        asked = {"questions_asked": state["questions_asked"] + 1}
        if choice == SKIP:
            return {**defer(state), **asked}
        return {**_with_current_updated(state, final_label=choice, source="human"), **asked}

    def accept(state: State) -> dict:
        # Reached when the model is trusted alone, or when the LLM agrees with it.
        segment = state["segments"][state["cursor"]]
        return _with_current_updated(state, final_label=segment["model_label"], source="model")

    def defer(state: State) -> dict:
        segment = state["segments"][state["cursor"]]
        if config.DEFERRED_LABEL == "llm" and segment.get("llm_label"):
            label, source = segment["llm_label"], "llm"
        else:
            label, source = segment["model_label"], "model"
        return _with_current_updated(state, final_label=label, source=source, needs_review=True)

    async def save_draft(state: State) -> dict:
        draft_segments = []
        for segment in state["segments"]:
            entry = {
                "start_sec": segment["start_sec"],
                "end_sec": segment["end_sec"],
                "class": segment["final_label"],
                "source": segment["source"],
            }
            entry.update({f: segment[f] for f in DRAFT_EXTRA_FIELDS if segment.get(f) not in (None, "")})
            draft_segments.append(entry)
        result = await client.save_draft_labels(state["video_id"], state["game_number"], draft_segments)
        return {"save_result": result}

    graph = StateGraph(State)
    graph.add_node("predict", predict)
    graph.add_node("next_segment", next_segment)
    graph.add_node("llm_opinion", llm_opinion)
    graph.add_node("human", human)
    graph.add_node("accept", accept)
    graph.add_node("defer", defer)
    graph.add_node("save_draft", save_draft)

    graph.add_edge(START, "predict")
    graph.add_edge("predict", "next_segment")
    graph.add_conditional_edges("next_segment", route, ["accept", "llm_opinion", "defer", "save_draft"])
    graph.add_conditional_edges("llm_opinion", route_llm, ["accept", "human", "defer"])
    graph.add_edge("human", "next_segment")
    graph.add_edge("accept", "next_segment")
    graph.add_edge("defer", "next_segment")
    graph.add_edge("save_draft", END)
    return graph.compile(checkpointer=checkpointer)
