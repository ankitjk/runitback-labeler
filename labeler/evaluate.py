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
import re
import sys
from collections import Counter
from pathlib import Path

from labeler import config, routing
from labeler.classes import CLASSES
from labeler.graph import frame_time
from labeler.guardrails import model_file_sha256
from labeler.mcp_client import LabelerClient

GRID_SEC = 0.1  # the labelling GUI rounds labels to 0.1 s
SWEEP_THRESHOLDS = [0.0] + [round(0.15 + 0.05 * i, 2) for i in range(15)]  # 0, then 0.15 ... 0.85


def unseen_games(model_dir: Path | None = None) -> frozenset:
    """The (video, game) pairs the model in this folder never trained on.

    The model is identified by the hash of its file. The table in config is
    checked first, so a listed model cannot be given other games by a file
    placed in its folder. Only an unlisted model is looked up in the run record
    RunItBack writes beside it (run.json, held-out games as "VIDEO:GAME").
    No model file, or a model in neither: no games, so nothing can be scored.
    """
    model_dir = Path(model_dir or config.MODEL_DIR)
    sha = model_file_sha256(model_dir)
    if sha is None:
        return frozenset()
    if sha in config.UNSEEN_GAMES_BY_MODEL:
        return config.UNSEEN_GAMES_BY_MODEL[sha]
    record = model_dir / "run.json"
    if not record.exists():
        return frozenset()
    games = set()
    for entry in json.loads(record.read_text())["split"]["test_games"]:
        video, _, number = entry.rpartition(":")
        games.add((video, int(number)))
    return frozenset(games)


def load_truth(video_id: str, game_number: int) -> list[dict]:
    if (video_id, game_number) not in unseen_games():
        raise ValueError(
            f"game {game_number} of {video_id} is not an unseen game for the model "
            f"{config.MODEL_NAME}; scoring it would give a number that means nothing"
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


def llm_accuracy(points: list[dict], segments: list[dict]) -> float:
    """Accuracy if every segment took the LLM's label. A segment with no usable LLM answer counts as wrong."""
    return sum(segments[p["segment"]].get("llm_label") == p["truth"] for p in points) / len(points) if points else 0.0


def _time_and_error(points: list[dict]) -> dict:
    wrong = sum(p["truth"] != p["model"] for p in points)
    return {
        "sec": round(len(points) * GRID_SEC, 1),
        "wrong_sec": round(wrong * GRID_SEC, 1),
        "error": wrong / len(points) if points else None,
    }


def _llm_verdict(segment: dict) -> str:
    """ "agree", "disagree" or "unusable", decided by the same routing function the graph uses."""
    if routing.after_llm(segment, 0, 1, False) == "accept":
        return "agree"
    # With the context agent switched on, only a real disagreement is sent to it.
    return "disagree" if routing.after_llm(segment, 0, 1, True) == "context" else "unusable"


def agreement(points: list[dict], segments: list[dict]) -> list[dict]:
    """Where the model and the LLM give the same class: how much time, and how much of it is wrong."""
    agreed = [p for p in points if _llm_verdict(segments[p["segment"]]) == "agree"]
    rows = [{"where": "all classes", **_time_and_error(agreed)}]
    for cls in sorted(config.NEVER_AUTO_ACCEPT):
        rows.append({"where": cls, **_time_and_error([p for p in agreed if p["model"] == cls])})
    return rows


def disagreement(points: list[dict], segments: list[dict]) -> dict:
    """Where both gave a class and the classes differ: who matches the human more often."""
    differ = [p for p in points if _llm_verdict(segments[p["segment"]]) == "disagree"]
    model_right = sum(p["truth"] == p["model"] for p in differ)
    llm_right = sum(p["truth"] == segments[p["segment"]]["llm_label"] for p in differ)
    return {
        "sec": round(len(differ) * GRID_SEC, 1),
        "model_right_sec": round(model_right * GRID_SEC, 1),
        "llm_right_sec": round(llm_right * GRID_SEC, 1),
        "neither_sec": round((len(differ) - model_right - llm_right) * GRID_SEC, 1),
    }


def sweep_with_llm(points: list[dict], segments: list[dict], thresholds: list[float]) -> list[dict]:
    """Like sweep(), with the LLM step added: a segment is also accepted when the LLM agrees with the model."""
    rows = []
    for threshold in thresholds:
        model_alone, agreed = set(), set()
        for i, s in enumerate(segments):
            if routing.after_predict(s, threshold, config.NEVER_AUTO_ACCEPT) == "accept":
                model_alone.add(i)
            elif routing.after_llm(s, 0, config.MAX_QUESTIONS, config.USE_CONTEXT_AGENT) == "accept":
                agreed.add(i)
        accepted_points = [p for p in points if p["segment"] in model_alone | agreed]
        result = _time_and_error(accepted_points)
        rows.append({
            "threshold": threshold,
            "accepted_model_alone": len(model_alone),
            "accepted_llm_agrees": len(agreed),
            "left_for_human": len(segments) - len(model_alone) - len(agreed),
            "coverage": len(accepted_points) / len(points),
            "accepted_sec": result["sec"],
            "wrong_sec": result["wrong_sec"],
            "auto_accept_error": result["error"],
        })
    return rows


def llm_report(points: list[dict], segments: list[dict], model_id: str, prompt_id: str) -> dict:
    return {
        "model_id": model_id,
        "prompt_id": prompt_id,
        "accuracy": llm_accuracy(points, segments),
        "zoom_segments": sum(bool(s.get("llm_zoom")) for s in segments),
        "failed_check_segments": sum(s.get("llm_error") is not None for s in segments),
        "agreement": agreement(points, segments),
        "disagreement": disagreement(points, segments),
        "sweep": sweep_with_llm(points, segments, SWEEP_THRESHOLDS),
    }


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
        f"# Evaluation: {report['video_id']} game {report['game_number']}",
        "",
        f"Compared every {GRID_SEC} s of the game with the human labels: "
        f"{report['labelled_sec']} s labelled, {report['segments']} model segments "
        f"(sampled at {report['sample_fps']} frames per second). "
        f"Scene model: {report.get('scene_model', 'not recorded')}.",
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
        f"second opinion, so even with no threshold not everything is accepted. "
        + (
            "The threshold for this model was chosen on this same game, so the error at the chosen threshold is optimistic."
            if report.get("scene_model_sha256") in config.THRESHOLD_BY_MODEL
            else "No threshold has been chosen for this model, so the pipeline accepts nothing on its word alone."
        ),
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
    if "llm" in report:
        lines += _llm_markdown(report["llm"], report["model_accuracy"], report["segmentation_ceiling"])
    return "\n".join(lines) + "\n"


def _llm_markdown(llm: dict, model_accuracy: float, ceiling: float) -> list[str]:
    d = llm["disagreement"]
    lines = [
        "",
        f"## With the first-opinion LLM ({llm['model_id']}, prompt {llm['prompt_id']})",
        "",
        "The LLM was shown the middle frame of every segment, once. Its class is worked out by code "
        "from the player states it reports.",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| LLM-only accuracy, all game time (every segment takes the LLM's class) | {_pct(llm['accuracy'])} |",
        f"| Model-only accuracy, for comparison | {_pct(model_accuracy)} |",
        f"| Most any labels could score with these segments (segmentation ceiling) | {_pct(ceiling)} |",
        f"| Segments the LLM flagged as a finishing-blow zoom | {llm['zoom_segments']} |",
        f"| Segments where the LLM's answer failed the checks | {llm['failed_check_segments']} |",
        "",
        "### When the model and the LLM agree",
        "",
        "| Class | Agreed (s) | Wrong (s) | Error |",
        "|---|---|---|---|",
    ]
    for row in llm["agreement"]:
        lines.append(f"| {row['where']} | {row['sec']} | {row['wrong_sec']} | {_pct(row['error'])} |")
    lines += [
        "",
        "### When they disagree",
        "",
        f"They gave different classes for {d['sec']} s. The model matched the human for "
        f"{d['model_right_sec']} s, the LLM for {d['llm_right_sec']} s, and neither for {d['neither_sec']} s.",
        "",
        "### Threshold sweep with the LLM",
        "",
        "Accepted = the model alone (as above), or the LLM asked and agreeing with the model. "
        "Everything else is left for the human.",
        "",
        "| Threshold | Accepted, model alone | Accepted, LLM agrees | Left for the human | Coverage | Accepted (s) | Wrong (s) | Auto-accept error |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in llm["sweep"]:
        name = "none" if row["threshold"] == 0 else f"{row['threshold']:.2f}"
        lines.append(
            f"| {name} | {row['accepted_model_alone']} | {row['accepted_llm_agrees']} | {row['left_for_human']} "
            f"| {_pct(row['coverage'])} | {row['accepted_sec']} | {row['wrong_sec']} | {_pct(row['auto_accept_error'])} |"
        )
    return lines


async def ask_llm_about_all(client, llm, video_id: str, segments: list[dict]) -> list[dict]:
    """Get the LLM's answer for every segment (cached after the first time)."""
    one_frame_at_a_time = asyncio.Lock()

    async def get_frame(t_sec: float) -> bytes:
        async with one_frame_at_a_time:
            return await client.get_frame(video_id, t_sec)

    async def ask(segment: dict) -> dict:
        t_sec = frame_time(segment)
        return {**segment, **await llm.label_frame(video_id, t_sec, lambda: get_frame(t_sec))}

    results = await asyncio.gather(*(ask(s) for s in segments), return_exceptions=True)
    failures = [r for r in results if isinstance(r, BaseException)]
    if failures:
        raise RuntimeError(
            f"{len(failures)} of {len(segments)} LLM answers are missing ({failures[0]}). "
            "The answers that did arrive are cached; run again to get the rest."
        )
    return results


async def evaluate(video_id: str, game_number: int, sample_fps: int | None = None, llm=None) -> dict:
    truth = load_truth(video_id, game_number)
    async with LabelerClient() as client:
        prediction = await client.predict_segments(video_id, game_number, sample_fps)
        segments = [{**s, "model_label": s["class"]} for s in prediction["segments"]]
        if llm is not None:
            segments = await ask_llm_about_all(client, llm, video_id, segments)
    points = build_points(truth, segments, prediction["start_sec"], prediction["end_sec"])
    with_llm = {"llm": llm_report(points, segments, llm.model, llm.prompt_id)} if llm is not None else {}
    return {
        **with_llm,
        "video_id": video_id,
        "game_number": game_number,
        "sample_fps": prediction["sample_fps"],
        "scene_model_sha256": prediction.get("scene_model_sha256"),
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
    parser.add_argument("--llm", action="store_true", help="also ask the LLM about every segment and score it")
    parser.add_argument("--llm-model", default=None, help="LLM model id (default: the project setting)")
    args = parser.parse_args()

    llm = None
    if args.llm or args.llm_model:
        from labeler.llm import GeminiLabeler
        llm = GeminiLabeler(model=args.llm_model)
    if not re.fullmatch(r"[A-Za-z0-9._-]+", config.MODEL_NAME):  # checked before any work or LLM call
        raise ValueError(f"model folder name {config.MODEL_NAME!r} cannot be used in a file name")
    report = asyncio.run(evaluate(args.video, args.game, args.fps, llm))
    report["scene_model"] = config.MODEL_NAME
    markdown = to_markdown(report)
    out_dir = config.OUTPUT_DIR / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    # The prompt id is in the file name, so a run with changed definitions writes a new
    # report beside the old one and never replaces it.
    # So is the scene model's name, so measuring a new model keeps the old model's reports.
    kind = f"llm_{llm.model}_{llm.prompt_id}" if llm is not None else "model_only"
    stem = f"{args.video}_g{args.game}_{report['sample_fps']}fps_{config.MODEL_NAME}_{kind}"
    (out_dir / f"{stem}.json").write_text(json.dumps(report, indent=2))
    (out_dir / f"{stem}.md").write_text(markdown)
    print(markdown)
    print(f"Saved to {out_dir / stem}.json and .md")
    if llm is not None:
        print(f"LLM ({llm.model}): {llm.calls_made} new calls ({llm.tokens_in} tokens in, {llm.tokens_out} out), the rest from the cache")
    return 0


if __name__ == "__main__":
    sys.exit(main())
