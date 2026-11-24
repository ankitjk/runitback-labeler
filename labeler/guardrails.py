"""Every check a draft must pass before it is written.

These are plain functions with no file or network access, so each rule can be
unit-tested. `save_draft_labels` in server.py calls them and refuses to write
if any rule is broken.
"""

import math
import re
from pathlib import Path

from labeler.classes import CLASSES, SOURCES

VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")

REQUIRED_FIELDS = ("start_sec", "end_sec", "class", "source")
OPTIONAL_FIELDS = (
    "needs_review",
    "model_label",
    "mean_confidence",
    "min_confidence",
    "llm_label",
    "llm_reason",
    "context_label",
    "context_reason",
    "context_frames",
)


def check_video_id(video_id: str) -> None:
    """A video id is 11 YouTube characters, so it can never be a file path."""
    if not isinstance(video_id, str) or not VIDEO_ID_PATTERN.fullmatch(video_id):
        raise ValueError(f"video_id {video_id!r} is not a valid id (11 letters, digits, '-' or '_')")


def draft_path(output_dir: Path, video_id: str, game_number: int) -> Path:
    """Where a draft is written. Refuses any path that would land outside output_dir."""
    check_video_id(video_id)
    if isinstance(game_number, bool) or not isinstance(game_number, int) or game_number < 1:
        raise ValueError(f"game_number {game_number!r} must be a whole number, 1 or more")
    root = output_dir.resolve()
    path = (root / f"{video_id}_g{game_number}_draft.json").resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"draft path {path} is outside {root}")
    return path


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _name(index: int, segment) -> str:
    if isinstance(segment, dict) and _is_number(segment.get("start_sec")) and _is_number(segment.get("end_sec")):
        return f"segment {index} ({segment['start_sec']:.2f}-{segment['end_sec']:.2f})"
    return f"segment {index}"


def validate_segments(
    segments: list,
    game_start: float,
    game_end: float,
    tolerance: float,
    existing_segments: list | None = None,
) -> list[str]:
    """Return every broken rule as a message. An empty list means the draft may be saved."""
    if not isinstance(segments, list) or not segments:
        return ["segments must be a non-empty list"]

    errors = []
    previous = None  # the last segment with usable times
    for index, segment in enumerate(segments):
        name = _name(index, segment)
        if not isinstance(segment, dict):
            errors.append(f"{name}: must be an object with start_sec, end_sec, class and source")
            continue

        for field in REQUIRED_FIELDS:
            if field not in segment:
                errors.append(f"{name}: missing '{field}'")
        for field in segment:
            if field not in REQUIRED_FIELDS and field not in OPTIONAL_FIELDS:
                errors.append(f"{name}: unknown field '{field}'")

        if "class" in segment and segment["class"] not in CLASSES:
            errors.append(f"{name}: class {segment['class']!r} is not one of the 11 classes")
        if "source" in segment and segment["source"] not in SOURCES:
            errors.append(f"{name}: source {segment['source']!r} is not one of {', '.join(SOURCES)}")
        if "needs_review" in segment and not isinstance(segment["needs_review"], bool):
            errors.append(f"{name}: needs_review must be true or false")

        start, end = segment.get("start_sec"), segment.get("end_sec")
        if not _is_number(start) or not _is_number(end):
            if "start_sec" in segment and "end_sec" in segment:
                errors.append(f"{name}: start_sec and end_sec must be numbers")
            continue

        if end <= start:
            errors.append(f"{name}: end_sec must be after start_sec")
        if start < game_start - tolerance or end > game_end + tolerance:
            errors.append(f"{name}: outside the game's bounds ({game_start:.2f}-{game_end:.2f})")
        if previous is not None:
            if start < previous["start_sec"]:
                errors.append(f"{name}: out of order, starts before the segment before it")
            elif start < previous["end_sec"]:
                errors.append(f"{name}: overlaps the segment before it, which ends at {previous['end_sec']:.2f}")
        previous = segment

    for old in existing_segments or []:
        if old.get("source") != "human":
            continue
        kept = any(new == old for new in segments)
        if not kept:
            errors.append(
                f"human-labelled segment ({old.get('start_sec')}-{old.get('end_sec')}, "
                f"{old.get('class')}) from the existing draft is changed or missing"
            )
    return errors
