"""Run the workflow on one game:  uv run python -m labeler.run --game 3

The run pauses whenever a segment needs a person. The question is printed
here, the frame is saved under output/review/ to look at, and the answer is
sent back into the graph, which carries on from where it stopped.
"""

import argparse
import asyncio
import sys
import uuid
from collections import Counter

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from labeler import config
from labeler.classes import CLASSES
from labeler.graph import SKIP, build_graph, human_choice
from labeler.llm import GeminiLabeler
from labeler.mcp_client import LabelerClient

# The graph loops once per segment, a few steps each. LangGraph stops a run
# after this many steps, which guards against a loop that never ends.
RECURSION_LIMIT = 10_000


def describe(segment: dict) -> str:
    asked = "llm_error" in segment
    if segment["source"] == "human":
        return "decided by the human"
    if not segment.get("needs_review"):
        return "accepted: model and LLM agree" if asked else "accepted: model alone"
    if not asked:
        return "deferred: no LLM was asked"
    if segment["llm_zoom"]:
        return "deferred: LLM saw a finishing-blow zoom"
    if segment["llm_label"] is None:
        return "deferred: LLM's answer failed the checks"
    return "deferred: model and LLM disagree"


def question_text(q: dict, frame_file) -> str:
    lines = [
        "",
        f"Question {q['question_number']} of at most {q['max_questions']}: "
        f"{q['start_sec']:.2f}-{q['end_sec']:.2f} s (frame at {q['t_sec']:.2f} s: {frame_file})",
        f"  1  {q['model_label']}  (model, confidence {q['min_confidence']:.2f})",
    ]
    if q["llm_label"]:
        lines.append(f"  2  {q['llm_label']}  (LLM: {q['llm_reason']})")
    elif q["llm_zoom"]:
        lines.append(f"     The LLM saw a finishing-blow zoom and gave no class ({q['llm_reason']})")
    else:
        lines.append("     The LLM gave no usable answer")
    lines.append("  s  skip (leave it marked for review)   or type a class name")
    return "\n".join(lines)


async def ask_person(client, q: dict) -> str:
    """Show one question and return an answer that is already known to be valid."""
    review_dir = config.OUTPUT_DIR / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    frame_file = review_dir / f"{q['video_id']}_{q['t_sec']:.2f}.jpg"
    frame_file.write_bytes(await client.get_frame(q["video_id"], q["t_sec"]))
    print(question_text(q, frame_file))
    while True:
        try:
            answer = await asyncio.to_thread(input, "> ")
        except EOFError:
            print("(no more input: skipping)")
            return SKIP
        if human_choice(answer, q) is not None:
            return answer
        print(f"Not a valid answer. Classes: {', '.join(CLASSES)}")


async def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-label one game and write a draft label file.")
    parser.add_argument("--video", default=config.DEFAULT_VIDEO_ID)
    parser.add_argument("--game", type=int, required=True)
    parser.add_argument("--llm-model", default=None, help="LLM model id (default: the project setting)")
    parser.add_argument("--no-llm", action="store_true", help="skip the LLM; unsure segments are deferred")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="most questions to ask the person (default: the project setting; 0 asks none)")
    args = parser.parse_args()

    llm = None if args.no_llm else GeminiLabeler(model=args.llm_model)
    # thread_id names this run, so the checkpointer knows which paused state to continue.
    run_config = {"recursion_limit": RECURSION_LIMIT, "configurable": {"thread_id": str(uuid.uuid4())}}
    async with LabelerClient() as client:
        graph = build_graph(client, llm=llm, checkpointer=InMemorySaver(), max_questions=args.max_questions)
        state = await graph.ainvoke({"video_id": args.video, "game_number": args.game}, config=run_config)
        while "__interrupt__" in state:  # the run is paused at a question
            answer = await ask_person(client, state["__interrupt__"][0].value)
            state = await graph.ainvoke(Command(resume=answer), config=run_config)

    segments = state["segments"]
    total_time = sum(s["end_sec"] - s["start_sec"] for s in segments)
    print(f"\n{len(segments)} segments, {total_time:.1f} s of video (threshold {config.CONFIDENCE_THRESHOLD})")
    counts, seconds = Counter(), Counter()
    for s in segments:
        counts[describe(s)] += 1
        seconds[describe(s)] += s["end_sec"] - s["start_sec"]
    for outcome, n in counts.most_common():
        print(f"  {outcome}: {n} segments, {seconds[outcome]:.1f} s ({seconds[outcome] / total_time:.0%} of the time)")

    if llm is not None:
        print(f"LLM ({llm.model}): {llm.calls_made} new calls ({llm.tokens_in} tokens in, {llm.tokens_out} out), the rest from the cache")
    result = state["save_result"]
    if not result["saved"]:
        print("Draft NOT saved:")
        for error in result["errors"]:
            print(f"  {error}")
        return 1
    print(f"Draft saved: {result['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
