"""Score the pipeline against the human labels for an unseen game.

    uv run python -m labeler.evaluate --game 3

This is the one module outside the MCP server that reads the human labels. It
is a measuring tool, not part of the pipeline: nothing it reads flows back
into a draft.
"""

import argparse
import asyncio
import json
import math
import sys
from collections import Counter

from labeler import config, routing
from labeler.classes import CLASSES
from labeler.mcp_client import LabelerClient

GRID_SEC = 0.1  # the labelling GUI rounds labels to 0.1 s
SWEEP_THRESHOLDS = [0.0] + [round(0.15 + 0.05 * i, 2) for i in range(15)]  # 0, then 0.15 ... 0.85


def load_truth(video_id: str, game_number: int) -> list[dict]:
    if (video_id, game_number) not in config.UNSEEN_GAMES:
        raise ValueError(
            f"game {game_number} of {video_id} is not an unseen game for this model; "
            "scoring it would give a number that means nothing"
        )
    truth = json.loads((config.GROUND_TRUTH_DIR / f"{video_id}.json").read_text())
    for game in truth["games"]:
        if game["game_number"] == game_number:
            return game["labels"]
    raise ValueError(f"no game {game_number} in the human labels for {video_id}")


def _label_at(labels: list[dict], t: float, key: str):
    """Return (index, label) of the segment covering time t, or (None, None)."""
    for index, label in enumerate(labels):
        if label["start_sec"] <= t < label["end_sec"]:
            return index, label[key]
    return None, None


def build_points(truth: list[dict], segments: list[dict], start: float, end: float) -> list[dict]:
    """One point per 0.1 s of the game that a human labelled: the true class and the model's segment."""
    points = []
    for tenth in range(math.ceil(start / GRID_SEC - 1e-9), math.floor(end / GRID_SEC + 1e-9)):
        t = (tenth + 0.5) * GRID_SEC
        _, true_class = _label_at(truth, t, "class")
        segment_index, model_label = _label_at(segments, t, "model_label")
        if true_class is None or segment_index is None:
            continue  # a moment no human labelled, or outside the model's segments
        points.append({"t": round(t, 2), "truth": true_class, "model": model_label, "segment": segment_index})
    return points


def accuracy(points: list[dict]) -> float:
    return sum(p["truth"] == p["model"] for p in points) / len(points) if points else 0.0


def per_class(points: list[dict]) -> list[dict]:
    rows = []
    for cls in CLASSES:
        true_n = sum(p["truth"] == cls for p in points)
        predicted_n = sum(p["model"] == cls for p in points)
        correct = sum(p["truth"] == cls and p["model"] == cls for p in points)
        if true_n == 0 and predicted_n == 0:
            continue
        rows.append({
            "class": cls,
            "true_sec": round(true_n * GRID_SEC, 1),
            "predicted_sec": round(predicted_n * GRID_SEC, 1),
            "recall": correct / true_n if true_n else None,
            "precision": correct / predicted_n if predicted_n else None,
        })
    return rows


def segmentation_ceiling(points: list[dict]) -> float:
    """Accuracy if every model segment were given its best possible single label."""
    by_segment: dict[int, Counter] = {}
    for p in points:
        by_segment.setdefault(p["segment"], Counter())[p["truth"]] += 1
    best = sum(counts.most_common(1)[0][1] for counts in by_segment.values())
    return best / len(points) if points else 0.0


def sweep(points: list[dict], segments: list[dict], thresholds: list[float]) -> list[dict]:
    """For each threshold: how much is accepted on the model's word, and how much of that is wrong."""
    rows = []
    for threshold in thresholds:
        accepted = {
            i for i, s in enumerate(segments)
            if routing.after_predict(s, threshold, config.NEVER_AUTO_ACCEPT) == "accept"
        }
        accepted_points = [p for p in points if p["segment"] in accepted]
        wrong = sum(p["truth"] != p["model"] for p in accepted_points)
        rows.append({
            "threshold": threshold,
            "accepted_segments": len(accepted),
            "second_opinion_segments": len(segments) - len(accepted),
            "coverage": len(accepted_points) / len(points),
            "accepted_sec": round(len(accepted_points) * GRID_SEC, 1),
            "wrong_sec": round(wrong * GRID_SEC, 1),
            "auto_accept_error": wrong / len(accepted_points) if accepted_points else None,
        })
    return rows


def sampled_frame_accuracy(truth: list[dict], prediction: dict, segments: list[dict]) -> dict:
    """Accuracy at the exact times the model looked at (comparable in kind to test_metrics.json)."""
    step = 1.0 / prediction["sample_fps"]
    hits = total = 0
    for i in range(prediction["n_frames"]):
        t = prediction["start_sec"] + (i + 0.5) * step
        _, true_class = _label_at(truth, t, "class")
        _, model_label = _label_at(segments, t, "model_label")
        if true_class is None:
            continue
        total += 1
        hits += true_class == model_label
    return {"frames": total, "correct": hits, "accuracy": hits / total if total else 0.0}


def _pct(value) -> str:
    return "-" if value is None else f"{value:.1%}"


def to_markdown(report: dict) -> str:
    lines = [
        f"# Model-only evaluation: {report['video_id']} game {report['game_number']}",
        "",
        f"Compared every {GRID_SEC} s of the game with the human labels: "
        f"{report['labelled_sec']} s labelled, {report['segments']} model segments "
        f"(sampled at {report['sample_fps']} frames per second).",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Model-only accuracy, all game time | {_pct(report['model_accuracy'])} |",
        f"| Model-only accuracy, at the {report['sampled']['frames']} sampled frames | {_pct(report['sampled']['accuracy'])} |",
        f"| Segmentation ceiling (best single label per model segment) | {_pct(report['segmentation_ceiling'])} |",
        "",
        "## Per class (by time)",
        "",
        "| Class | True (s) | Predicted (s) | Recall | Precision |",
        "|---|---|---|---|---|",
    ]
    for row in report["per_class"]:
        lines.append(f"| {row['class']} | {row['true_sec']} | {row['predicted_sec']} | {_pct(row['recall'])} | {_pct(row['precision'])} |")
    never = ", ".join(sorted(config.NEVER_AUTO_ACCEPT))
    lines += [
        "",
        "## Threshold sweep",
        "",
        f"A segment is accepted on the model's word if its lowest frame confidence is at or above "
        f"the threshold and its label is not one of: {never}. Those two classes always go for a "
        f"second opinion, so even with no threshold not everything is accepted. The threshold was "
        f"chosen on this same game, so the error at the chosen threshold is optimistic.",
        "",
        "| Threshold | Accepted segments | Sent for a second opinion | Coverage (share of time accepted) | Accepted (s) | Wrong (s) | Auto-accept error |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in report["sweep"]:
        name = "none" if row["threshold"] == 0 else f"{row['threshold']:.2f}"
        lines.append(
            f"| {name} | {row['accepted_segments']} | {row['second_opinion_segments']} | {_pct(row['coverage'])} "
            f"| {row['accepted_sec']} | {row['wrong_sec']} | {_pct(row['auto_accept_error'])} |"
        )
    return "\n".join(lines) + "\n"


async def evaluate(video_id: str, game_number: int, sample_fps: int | None = None) -> dict:
    truth = load_truth(video_id, game_number)
    async with LabelerClient() as client:
        prediction = await client.predict_segments(video_id, game_number, sample_fps)
    segments = [{**s, "model_label": s["class"]} for s in prediction["segments"]]
    points = build_points(truth, segments, prediction["start_sec"], prediction["end_sec"])
    return {
        "video_id": video_id,
        "game_number": game_number,
        "sample_fps": prediction["sample_fps"],
        "segments": len(segments),
        "labelled_sec": round(len(points) * GRID_SEC, 1),
        "model_accuracy": accuracy(points),
        "sampled": sampled_frame_accuracy(truth, prediction, segments),
        "segmentation_ceiling": segmentation_ceiling(points),
        "per_class": per_class(points),
        "sweep": sweep(points, segments, SWEEP_THRESHOLDS),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Score the pipeline against the human labels for an unseen game.")
    parser.add_argument("--video", default=config.DEFAULT_VIDEO_ID)
    parser.add_argument("--game", type=int, required=True)
    parser.add_argument("--fps", type=int, default=None, help="frames sampled per second (default: the project setting)")
    args = parser.parse_args()

    report = asyncio.run(evaluate(args.video, args.game, args.fps))
    markdown = to_markdown(report)
    out_dir = config.OUTPUT_DIR / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.video}_g{args.game}_{report['sample_fps']}fps_model_only"
    (out_dir / f"{stem}.json").write_text(json.dumps(report, indent=2))
    (out_dir / f"{stem}.md").write_text(markdown)
    print(markdown)
    print(f"Saved to {out_dir / stem}.json and .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
