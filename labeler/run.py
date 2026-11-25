"""Run the workflow on one game:  uv run python -m labeler.run --game 3"""

import argparse
import asyncio
import sys
from collections import Counter

from labeler import config
from labeler.graph import build_graph
from labeler.llm import GeminiLabeler
from labeler.mcp_client import LabelerClient

# The graph loops once per segment, a few steps each. LangGraph stops a run
# after this many steps, which guards against a loop that never ends.
RECURSION_LIMIT = 10_000


def describe(segment: dict) -> str:
    asked = "llm_error" in segment
    if not segment.get("needs_review"):
        return "accepted: model and LLM agree" if asked else "accepted: model alone"
    if not asked:
        return "deferred: no LLM was asked"
    if segment["llm_zoom"]:
        return "deferred: LLM saw a finishing-blow zoom"
    if segment["llm_label"] is None:
        return "deferred: LLM's answer failed the checks"
    return "deferred: model and LLM disagree"


async def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-label one game and write a draft label file.")
    parser.add_argument("--video", default=config.DEFAULT_VIDEO_ID)
    parser.add_argument("--game", type=int, required=True)
    parser.add_argument("--llm-model", default=None, help="LLM model id (default: the project setting)")
    parser.add_argument("--no-llm", action="store_true", help="skip the LLM; unsure segments are deferred")
    args = parser.parse_args()

    llm = None if args.no_llm else GeminiLabeler(model=args.llm_model)
    async with LabelerClient() as client:
        graph = build_graph(client, llm=llm)
        state = await graph.ainvoke(
            {"video_id": args.video, "game_number": args.game},
            config={"recursion_limit": RECURSION_LIMIT},
        )

    segments = state["segments"]
    total_time = sum(s["end_sec"] - s["start_sec"] for s in segments)
    print(f"{len(segments)} segments, {total_time:.1f} s of video (threshold {config.CONFIDENCE_THRESHOLD})")
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
