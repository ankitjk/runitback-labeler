"""Settings for the whole pipeline. Change behaviour here, not in the code."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# RunItBack is read-only from this repo.
RUNITBACK_ROOT = Path(os.environ.get("RUNITBACK_ROOT", "~/Projects/RunItBack")).expanduser()
VIDEO_CACHE_DIR = RUNITBACK_ROOT / ".video_cache"
GROUND_TRUTH_DIR = RUNITBACK_ROOT / "data" / "scene_classifier" / "raw_labels"

# The scene model's folder: scene_classifier.onnx and class_mapping.json. Set
# LABELER_MODEL_DIR to use another model, for example a run folder written by RunItBack.
MODEL_DIR = Path(os.environ.get("LABELER_MODEL_DIR") or REPO_ROOT / "models" / "heldout_g3").expanduser().resolve()
MODEL_NAME = MODEL_DIR.name  # goes into report file names
OUTPUT_DIR = Path(os.environ.get("LABELER_OUTPUT_DIR", REPO_ROOT / "output")).expanduser()

DEFAULT_VIDEO_ID = "XUkRx6CpL04"

# Models are known by the SHA-256 of scene_classifier.onnx, never by folder name.
HELDOUT_G3_SHA256 = "e55632bd4f704d578b13e7f171c2a47563cd47adba10f42a1e7e28a7a31d9e96"

# The games a model never trained on. The evaluation refuses any other game,
# because scores on training games are not real scores. This table is checked
# first. Only a model that is not in it is looked up in the run record that
# RunItBack writes beside a model (run.json). A model in neither has no unseen games.
UNSEEN_GAMES_BY_MODEL = {HELDOUT_G3_SHA256: frozenset({("XUkRx6CpL04", 3)})}

SAMPLE_FPS = 2

# Chosen from the game-3 sweep at 2 fps (output/eval/, measured 2026-11-24): the lowest
# threshold whose auto-accepted time is under 10% wrong. At 0.40 the model alone
# is trusted for 11 of 218 segments (15.0% of game time, 5.4% of it wrong); at
# 0.35 the error is 12.7%. Chosen and measured on the same game, so optimistic.
CONFIDENCE_THRESHOLD = 0.40

# A threshold holds only for the model it was measured on. A model that is not
# listed here has no threshold, so nothing is accepted on its word alone.
# (The 2026-11-26 RunItBack baseline at 0.40: 8 segments accepted, 44.4% of that time wrong.)
THRESHOLD_BY_MODEL = {HELDOUT_G3_SHA256: CONFIDENCE_THRESHOLD}

# The model predicts these about 5x too often, so its word alone is never enough.
NEVER_AUTO_ACCEPT = frozenset({"respawning", "one_on_ledge"})

# The first-opinion LLM. The key is read from this environment variable and nowhere else.
# Measured on game 3 (2026-11-24): gemini-4-hydrogen's labels match the human for
# 69.2% of game time, gemini-3.5-flash-lite's for 50.5%. Hydrogen needs a paid
# Google tier: the free tier gave it 20 requests a day, and one game needs about
# 200. gemini-3.5-flash-lite ran a whole game on the free tier (--llm-model).
LLM_MODEL = "gemini-4-hydrogen"
LLM_API_KEY_ENV = "GEMINI_API_KEY"
LLM_CONCURRENCY = 4  # calls in flight at once

MAX_QUESTIONS = 10

# The ground truth's own first game-3 label starts 0.017 s before the game does.
BOUNDS_TOLERANCE_SEC = 0.05

# Which label a segment gets when it was bound for the human but the question
# cap was reached: "model" or "llm". Measured on game 3 with gemini-4-hydrogen:
# where the two disagreed (120.0 s) the LLM matched the human for 78.1 s and the
# model for 6.1 s, so the LLM's label is used when it gave one.
DEFERRED_LABEL = "llm"

USE_CONTEXT_AGENT = False
CONTEXT_MAX_FRAMES = 4
CONTEXT_WINDOW_SEC = 2.0
