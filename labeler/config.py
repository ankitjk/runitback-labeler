"""Settings for the whole pipeline. Change behaviour here, not in the code."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# RunItBack is read-only from this repo.
RUNITBACK_ROOT = Path(os.environ.get("RUNITBACK_ROOT", "~/Projects/RunItBack")).expanduser()
VIDEO_CACHE_DIR = RUNITBACK_ROOT / ".video_cache"
GROUND_TRUTH_DIR = RUNITBACK_ROOT / "data" / "scene_classifier" / "raw_labels"

MODEL_DIR = REPO_ROOT / "models" / "heldout_g3"
OUTPUT_DIR = Path(os.environ.get("LABELER_OUTPUT_DIR", REPO_ROOT / "output")).expanduser()

DEFAULT_VIDEO_ID = "XUkRx6CpL04"

# The only games the model in MODEL_DIR never trained on. The evaluation
# refuses any other game, because scores on training games are not real scores.
UNSEEN_GAMES = frozenset({("XUkRx6CpL04", 3)})

SAMPLE_FPS = 2

# Chosen from the game-3 sweep at 2 fps (output/eval/, measured 2026-11-24): the lowest
# threshold whose auto-accepted time is under 10% wrong. At 0.40 the model alone
# is trusted for 11 of 218 segments (15.0% of game time, 5.4% of it wrong); at
# 0.35 the error is 12.7%. Chosen and measured on the same game, so optimistic.
CONFIDENCE_THRESHOLD = 0.40

# The model predicts these about 5x too often, so its word alone is never enough.
NEVER_AUTO_ACCEPT = frozenset({"respawning", "one_on_ledge"})

MAX_QUESTIONS = 10

# The ground truth's own first game-3 label starts 0.017 s before the game does.
BOUNDS_TOLERANCE_SEC = 0.05

# Which label a segment gets when it was bound for the human but the question
# cap was reached: "model" or "llm".
DEFERRED_LABEL = "model"

USE_CONTEXT_AGENT = False
CONTEXT_MAX_FRAMES = 4
CONTEXT_WINDOW_SEC = 2.0
