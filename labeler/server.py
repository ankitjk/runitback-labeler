"""The MCP server: the only code that touches the video, the model and label files.

Run it with:  uv run python -m labeler.server
It talks over stdin/stdout, so nothing here may print to stdout.
"""

import json
import os
import tempfile
from pathlib import Path

import cv2
from mcp.server.fastmcp import FastMCP, Image

from labeler import config
from labeler.guardrails import check_video_id, draft_path, validate_segments
from labeler.inference import SceneModel

mcp = FastMCP("runitback-labeler", log_level="WARNING")

BATCH_SIZE = 32
_model: SceneModel | None = None


def _get_model() -> SceneModel:
    global _model
    if _model is None:
        _model = SceneModel(config.MODEL_DIR)
    return _model


def _video_path(video_id: str) -> Path:
    check_video_id(video_id)
    path = config.VIDEO_CACHE_DIR / f"{video_id}.mp4"
    if not path.is_file():
        raise ValueError(f"no video for {video_id!r} in the video cache")
    return path


def game_bounds(video_id: str, game_number: int) -> dict:
    """A game's start, end and stage. Reads the ground-truth file but never its labels."""
    check_video_id(video_id)
    truth_file = config.GROUND_TRUTH_DIR / f"{video_id}.json"
    if not truth_file.is_file():
        raise ValueError(f"no game list for {video_id!r}")
    truth = json.loads(truth_file.read_text())
    for game in truth.get("games", []):
        if game.get("game_number") == game_number:
            return {
                "video_path": truth.get("video_path", f".video_cache/{video_id}.mp4"),
                "stage": game.get("stage", "unknown"),
                "start_sec": float(game["start_sec"]),
                "end_sec": float(game["end_sec"]),
            }
    raise ValueError(f"video {video_id!r} has no game {game_number!r}")


def _sample_times(start: float, end: float, fps: float) -> list[float]:
    """The middle of each 1/fps-second cell between start and end."""
    step = 1.0 / fps
    times = []
    t = start + step / 2
    while t < end:
        times.append(t)
        t += step
    return times


def _read_frames(video: Path, times: list[float]):
    """Yield (time, frame) reading forward through the video, which is faster than seeking each time."""
    cap = cv2.VideoCapture(str(video))
    try:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        targets = [round(t * video_fps) for t in times]
        cap.set(cv2.CAP_PROP_POS_FRAMES, targets[0])
        position = targets[0]
        for t, target in zip(times, targets):
            while position < target:
                if not cap.grab():
                    return
                position += 1
            ok, frame = cap.read()
            position += 1
            if not ok:
                return
            yield t, frame
    finally:
        cap.release()


def _merge(times: list[float], predictions: list[tuple[str, float]], start: float, end: float, fps: float) -> list[dict]:
    """Join neighbouring frames with the same class into segments."""
    step = 1.0 / fps
    segments = []
    for i, ((cls, confidence), _t) in enumerate(zip(predictions, times)):
        cell_start = start + i * step
        cell_end = min(start + (i + 1) * step, end)
        if segments and segments[-1]["class"] == cls:
            segments[-1]["end_sec"] = cell_end
            segments[-1]["_confidences"].append(confidence)
        else:
            segments.append({"class": cls, "start_sec": cell_start, "end_sec": cell_end, "_confidences": [confidence]})
    if segments:
        segments[-1]["end_sec"] = end
    for segment in segments:
        confidences = segment.pop("_confidences")
        segment["start_sec"] = round(segment["start_sec"], 2)
        segment["end_sec"] = round(segment["end_sec"], 2)
        segment["mean_confidence"] = round(sum(confidences) / len(confidences), 4)
        segment["min_confidence"] = round(min(confidences), 4)
        segment["n_frames"] = len(confidences)
    return segments


def _write_json(path: Path, data: dict) -> None:
    """Write to a temporary file, then rename, so a crash cannot leave half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


@mcp.tool()
def predict_segments(video_id: str, game_number: int) -> dict:
    """Label one game with the small scene model and return its segments.

    Samples the game a few times per second, classifies each frame, and merges
    neighbouring frames with the same class. Each segment has class, start_sec,
    end_sec, mean_confidence, min_confidence and n_frames.
    """
    video = _video_path(video_id)
    bounds = game_bounds(video_id, game_number)
    model = _get_model()
    fps = config.SAMPLE_FPS

    cache_file = config.OUTPUT_DIR / "cache" / f"predictions_{video_id}_g{game_number}_{fps}fps_{model.fingerprint}.json"
    if cache_file.is_file():
        return json.loads(cache_file.read_text())

    times = _sample_times(bounds["start_sec"], bounds["end_sec"], fps)
    predictions, batch = [], []
    for _t, frame in _read_frames(video, times):
        batch.append(frame)
        if len(batch) == BATCH_SIZE:
            predictions.extend(model.predict(batch))
            batch = []
    if batch:
        predictions.extend(model.predict(batch))
    if len(predictions) != len(times):
        raise RuntimeError(f"read {len(predictions)} of {len(times)} frames from the video")

    result = {
        "video_id": video_id,
        "game_number": game_number,
        "stage": bounds["stage"],
        "start_sec": bounds["start_sec"],
        "end_sec": bounds["end_sec"],
        "sample_fps": fps,
        "n_frames": len(times),
        "segments": _merge(times, predictions, bounds["start_sec"], bounds["end_sec"], fps),
    }
    _write_json(cache_file, result)
    return result


@mcp.tool()
def get_frame(video_id: str, t_sec: float, max_width: int = 1280) -> Image:
    """Return the video frame at t_sec (seconds from the start of the video) as a JPEG."""
    video = _video_path(video_id)
    if not 64 <= max_width <= 1920:
        raise ValueError("max_width must be between 64 and 1920")
    cap = cv2.VideoCapture(str(video))
    try:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / video_fps
        if not 0 <= t_sec < duration:
            raise ValueError(f"t_sec {t_sec} is outside the video (0-{duration:.1f} s)")
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(t_sec * video_fps))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise RuntimeError(f"could not read the frame at {t_sec} s")
    height, width = frame.shape[:2]
    if width > max_width:
        frame = cv2.resize(frame, (max_width, round(height * max_width / width)), interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("could not encode the frame")
    return Image(data=jpeg.tobytes(), format="jpeg")


@mcp.tool()
def save_draft_labels(video_id: str, game_number: int, segments: list[dict]) -> dict:
    """Check a game's draft labels and, only if every check passes, write them under output/.

    Each segment needs start_sec, end_sec, class (one of the 11 scene classes)
    and source (model, llm or human). Returns {"saved": true, "path": ...} or
    {"saved": false, "errors": [...]} listing every broken rule.
    """
    _video_path(video_id)
    path = draft_path(config.OUTPUT_DIR, video_id, game_number)
    bounds = game_bounds(video_id, game_number)

    existing_segments = []
    if path.is_file():
        for game in json.loads(path.read_text()).get("games", []):
            if game.get("game_number") == game_number:
                existing_segments = game.get("labels", [])

    errors = validate_segments(
        segments, bounds["start_sec"], bounds["end_sec"], config.BOUNDS_TOLERANCE_SEC, existing_segments
    )
    if errors:
        return {"saved": False, "errors": errors}

    draft = {
        "video_id": video_id,
        "video_path": bounds["video_path"],
        "games": [
            {
                "game_number": game_number,
                "stage": bounds["stage"],
                "start_sec": bounds["start_sec"],
                "end_sec": bounds["end_sec"],
                "labels": segments,
            }
        ],
    }
    _write_json(path, draft)
    return {"saved": True, "path": str(path), "n_segments": len(segments)}


if __name__ == "__main__":
    mcp.run()
