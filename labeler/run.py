"""Run the workflow on one game:  uv run python -m labeler.run --game 3"""

import argparse
import asyncio
import sys
from collections import Counter

from labeler import config
from labeler.graph import build_graph
from labeler.mcp_client import LabelerClient

# The graph loops once per segment, a few steps each. LangGraph stops a run
# after this many steps, which guards against a loop that never ends.
RECURSION_LIMIT = 10_000


def describe(segment: dict) -> str:
    if segment.get("needs_review"):
        return "deferred (needs review)"
    return f"accepted from {segment['source']}"


async def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-label one game and write a draft label file.")
    parser.add_argument("--video", default=config.DEFAULT_VIDEO_ID)
    parser.add_argument("--game", type=int, required=True)
    args = parser.parse_args()

    async with LabelerClient() as client:
        graph = build_graph(client)
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
