# The draft label file

What this pipeline writes, how RunItBack's labelling GUI reads it, and what the
GUI still needs before a person can correct a draft in it.

## Where it goes

`output/<video_id>_g<game_number>_draft.json`, for example
`output/XUkRx6CpL04_g3_draft.json`. Only the MCP tool `save_draft_labels`
writes it, and only inside `output/`. Nothing here ever writes to RunItBack's
`data/scene_classifier/raw_labels/`.

## Format

The same structure as RunItBack's hand-label files, with extra fields on each
segment:

```json
{
  "video_id": "XUkRx6CpL04",
  "video_path": ".video_cache/XUkRx6CpL04.mp4",
  "games": [
    {
      "game_number": 3,
      "stage": "final_destination",
      "start_sec": 437.12,
      "end_sec": 646.4,
      "labels": [
        {"start_sec": 437.12, "end_sec": 465.5, "class": "non_gameplay",
         "source": "model", "model_label": "non_gameplay",
         "mean_confidence": 0.93, "min_confidence": 0.91},

        {"start_sec": 512.0, "end_sec": 513.5, "class": "one_offstage",
         "source": "human", "model_label": "one_on_ledge",
         "mean_confidence": 0.47, "min_confidence": 0.41,
         "llm_label": "one_offstage",
         "llm_reason": "One player is below the stage edge, not holding the ledge"},

        {"start_sec": 530.0, "end_sec": 530.5, "class": "one_airborne",
         "source": "model", "needs_review": true, "model_label": "one_airborne",
         "mean_confidence": 0.38, "min_confidence": 0.38,
         "llm_label": "both_airborne", "llm_reason": "Both players are off the ground"}
      ]
    }
  ]
}
```

(The values above are made up to show the shape.)

### Fields the GUI needs

| Field | Meaning |
|---|---|
| `start_sec`, `end_sec` | The segment's time range, in seconds from the start of the video |
| `class` | One of the 11 scene classes in `CLASSES.md` |

### Fields this pipeline adds

| Field | Meaning |
|---|---|
| `source` | Who decided `class`: `model` (alone, or agreeing with the first LLM), `llm` (the context agent settled a disagreement, or a deferred segment given the LLM's label) or `human`. Always present |
| `needs_review` | `true` if the segment was bound for the human but the question cap was reached. Absent otherwise |
| `model_label` | The small model's label |
| `mean_confidence`, `min_confidence` | The model's average and lowest confidence over the segment's frames |
| `llm_label`, `llm_reason` | The first LLM's class and its one-line reason. Present only if it was asked |
| `context_label`, `context_reason`, `context_frames` | The context agent's class, its reason, and the times of the frames it chose to look at. Present only if it was asked |

`stage`, and the game's `start_sec` and `end_sec`, are copied from the
ground-truth file's entry for that game.

## Checks before a draft is written

Done in code inside `save_draft_labels` (`labeler/guardrails.py`). If any check
fails, nothing is written and the error names the segment and the rule.

- every `class` is one of the 11 (`finishing_blow` is refused: the GUI cannot load it)
- every `source` is `model`, `llm` or `human`
- every segment ends after it starts and lies inside the game's bounds (0.05 s tolerance)
- segments are sorted and do not overlap
- a segment already saved with `source: human` cannot be changed or removed by a later save
- the file's path resolves to inside `output/`

## Opening a draft in the GUI

```
cd ~/Projects/RunItBack
python tools/scene_labeler.py .video_cache/XUkRx6CpL04.mp4 \
    --labels ~/Projects/runitback-labeler/output/XUkRx6CpL04_g3_draft.json
```

The GUI reads `start_sec`, `end_sec` and `class` from each segment and ignores
the other fields (`VideoLabels.from_dict`, `tools/scene_labeler.py:305-331`).
The segments appear in its table, and clicking a row moves the video to that
segment. Ctrl+S saves to the path given by `--labels`, so it saves over the
draft, not over the ground truth.

## What the GUI cannot do yet

Checked against `tools/scene_labeler.py` on 2026-11-24. The GUI was built for
labelling from nothing, not for correcting.

1. **A loaded segment cannot be changed or deleted.** The GUI can add a label
   (number keys, `R` for respawning) and undo labels added in the current
   session. Labelling over a wrong segment adds a second, overlapping segment
   and leaves the wrong one in place.
2. **Saving drops the extra fields.** `VideoLabels.to_dict`
   (`tools/scene_labeler.py:282-303`) writes only `start_sec`, `end_sec` and
   `class`. After one Ctrl+S, `source`, the confidences and the LLM's reason
   are gone.
3. **Nothing shows which segments need checking.** `needs_review` and `source`
   are not displayed.

Until these are fixed, corrections happen in this pipeline's terminal review,
and the draft is opened in the GUI to look at, not to edit.

## Changes needed in RunItBack (not done from this repo)

In `tools/scene_labeler.py`:

1. Keep unknown per-segment fields on load and write them back on save.
2. Let the user select a segment and change its class or delete it. A changed
   segment gets `source: human` and loses `needs_review`.
3. When a new label overlaps existing ones, trim or replace them so labels
   never overlap.
4. Show `source` and `needs_review` in the table (a column or a colour), with a
   key to jump to the next segment that needs review.

After that, the finished file is copied into `raw_labels/` by the person, as a
deliberate step. Training reads only `start_sec`, `end_sec` and `class`, so the
extra fields do no harm there.
