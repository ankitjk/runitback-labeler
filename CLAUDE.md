# runitback-labeler

An agentic pre-labelling pipeline for RunItBack's scene classifier. It proposes
scene labels for Super Smash Bros. Ultimate gameplay, routes uncertain or
disputed segments to a human, and writes a draft label file that the human
finishes in RunItBack's existing labelling tool.

Built with LangGraph (the workflow) and an MCP server (the only access to data).

| Read this | For |
|-----------|-----|
| `PLAN.md` | Build steps, settled decisions, module layout |
| `CLASSES.md` | Class definitions and the ranking |
| `docs/DRAFT_FORMAT.md` | The draft file's fields, and what the labelling GUI can and cannot do with it |
| `docs/CONCEPTS.md` | LangGraph, MCP and evaluation terms in plain words |

Run the `rules-reviewer` agent after finishing a plan step and before any commit.

## Why it exists

RunItBack's bottleneck is labelled data. The whole corpus is one video
(`XUkRx6CpL04`, 3 games, ~1,580 frames). On a game it has never seen, the scene
model scores 48.3% (pooled over 3 leave-one-game-out folds). The 92% from
earlier came from a random frame split that leaked near-duplicate frames 0.5 s
apart, so it is not a real score. More labelled videos are the fix, and hand
labelling is slow.

## Relationship to RunItBack

RunItBack lives at `~/Projects/RunItBack`. This repo talks to it **through
files only**, never by importing its code:

| What | Path |
|------|------|
| Human labels (ground truth) | `~/Projects/RunItBack/data/scene_classifier/raw_labels/<video_id>.json` |
| Video | `~/Projects/RunItBack/.video_cache/XUkRx6CpL04.mp4` (~470 MB) |
| Labelling GUI | `~/Projects/RunItBack/tools/scene_labeler.py` - opens a draft with `--labels <file>` |
| Model used here | `models/heldout_g3/scene_classifier.onnx` + `class_mapping.json` (this repo, gitignored) |

Do not modify RunItBack from this repo. Its ground-truth label files are read-only here.

## Label format (must match exactly - the GUI loads it)

```json
{
  "video_id": "XUkRx6CpL04",
  "video_path": ".video_cache/XUkRx6CpL04.mp4",
  "games": [
    {"game_number": 3, "stage": "final_destination",
     "start_sec": 437.12, "end_sec": 646.4,
     "labels": [{"start_sec": 437.12, "end_sec": 451.0, "class": "non_gameplay"}, ...]}
  ]
}
```

Labels are **time segments**, not frames. Human segments are short, mostly
0.2-0.6 s. Game bounds in the ground-truth file: game 1 0.0-204.0 s
(smashville), game 2 204.0-437.1 s (battlefield), game 3 437.1-646.4 s
(final_destination).

Facts checked on 2026-11-24:

- The video is 1920x1080 at 60 frames per second, 38,785 frames.
- Game 3's first human label starts at 437.1 s, just before the game's
  `start_sec` (437.1167). Bounds checks need a small tolerance (0.05 s).
- The human labels have small gaps and overlaps: game 3 has a 0.1 s gap at
  493.4-493.5 s; game 1 has a 0.1 s overlap at 84.5-84.6 s.
- The labelling GUI loads a draft and ignores extra fields, but it cannot
  change or delete a loaded segment, and saving drops the extra fields
  (including `source`). Details in `docs/DRAFT_FORMAT.md`.

## The 11 scene classes

`non_gameplay, both_grounded, one_airborne, both_airborne, one_offstage,
both_offstage, one_on_ledge, both_on_ledge, one_knocked_down,
both_knocked_down, respawning`

**Definitions, tie-break ranking and open questions: `CLASSES.md`.** Use it
for the LLM prompt and the review screen; don't paraphrase it from memory.

The model knows only the classes that had enough training frames. **Always
read its output order from `class_mapping.json`** (`{"0": "both_airborne", ...}`).
Never hardcode it: RunItBack has an open bug (ISSUE-017) from doing exactly that.

## Model inference (match RunItBack exactly)

From `~/Projects/RunItBack/runitback/cv/scene_classifier.py:207-220`, per frame
(OpenCV BGR):

1. `cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)`
2. BGR -> RGB
3. `/ 255.0`, then ImageNet normalise: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`
4. HWC -> NCHW float32, run with onnxruntime, softmax the output for confidence

Copy these lines. Don't import RunItBack.

## The model in `models/heldout_g3/`

Trained on games 1-2 of `XUkRx6CpL04` only (trained 2026-11-24, best epoch 10).
**Game 3 was never seen**, so game 3 is the honest evaluation set for this whole
pipeline. Never evaluate on games 1 or 2 with this model: it trained on them.

It outputs **8 classes**, in this order (also in `class_mapping.json`):
`both_airborne, both_grounded, both_offstage, non_gameplay, one_airborne,
one_offstage, one_on_ledge, respawning`. It **cannot** predict `both_on_ledge`,
`one_knocked_down` or `both_knocked_down` (too few training frames), so those
can only come from the LLM or the human.

Measured on game 3 (490 frames, `test_metrics.json`): **39.2% accuracy**, 44.9%
macro recall.

| Class | True frames | Predicted | Recall | Precision |
|---|---|---|---|---|
| both_airborne | 45 | 3 | 4.4% | 66.7% |
| both_grounded | 49 | 32 | 46.9% | 71.9% |
| both_offstage | 82 | 44 | 41.5% | 77.3% |
| non_gameplay | 84 | 118 | 78.6% | 55.9% |
| one_airborne | 101 | 35 | 17.8% | 51.4% |
| one_offstage | 91 | 52 | 19.8% | 34.6% |
| one_on_ledge | 14 | 79 | 50.0% | 8.9% |
| respawning | 24 | 127 | 100% | 18.9% |

These 490 frames were taken from inside each labelled segment, at least 0.15 s
from its edges (`tools/extract_labeled_frames.py` in RunItBack). This
pipeline's own evaluation covers all game time, including scene changes, so
its model-only accuracy is a different number. Say which is which.

It badly over-predicts the rare classes `respawning` and `one_on_ledge` (5x
their true count), an effect of inverse-frequency class weights. So a
`respawning` or `one_on_ledge` prediction should never be auto-accepted, and
confidence thresholds must be **measured** on game 3, not guessed. The model
being weak is the point of the project: most segments will need a second
opinion or a human, and the evaluation shows how many.

`scene_classifier.onnx` takes input `input`, shape `(N, 3, 224, 224)` float32,
any batch size N, and outputs `output`, shape `(N, 8)` raw logits. It was
exported with PyTorch's older exporter (`dynamo=False`), because the default
exporter in torch 2.13 fixed the batch size at 1. It was checked against
PyTorch: max difference 2e-6 for batches of 1 and 5. `checkpoint_*.pt` files
are training checkpoints, not needed here.

## Environment

- Python 3.12 through `uv` (the system Python is 3.15). Run things with `uv run`.
- Pin `mcp>=1.30,<2`. Version 2.x renamed FastMCP to MCPServer, and
  `langchain-mcp-adapters` 0.3.1 fails to import with it.

## Rules

- **Guardrails live in code** (inside the MCP tools), never only in a prompt.
- **Never overwrite a human-labelled segment.** Drafts go to new files under
  `output/`, never over RunItBack's `raw_labels/`.
- **Every segment records where it came from** (`source`: `model`, `llm` or
  `human`). This is the minimal audit trail.
- **Secrets:** API keys come from environment variables only (macOS Keychain ->
  `~/.zprofile`). Never write a key into a file in this repo, including `.env`
  if it could be committed. A key committed by mistake stays in git history for good.
- **Never commit** videos, extracted frames, model files or `output/`. They are gitignored.
- **Be honest about numbers.** Report measured results only, and say when
  something wasn't measured. This project will be shown publicly.
