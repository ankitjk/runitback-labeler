# runitback-labeler

A pipeline that drafts scene labels for Super Smash Bros. Ultimate gameplay
video, so a person corrects a draft instead of labelling from nothing. A small
vision model labels every frame, an LLM gives a second opinion where the model
can't be trusted, and a person is asked about the segments the two disagree
on. Every label in the draft records who decided it.

It is built with [LangGraph](https://langchain-ai.github.io/langgraph/) (the
workflow) and an [MCP](https://modelcontextprotocol.io/) server (the only
access to the video, the model and the label files).

**Status:** a working first demo, measured on one game of one video. The
numbers below are small-sample results, not claims about other videos.

## The problem

[RunItBack](https://github.com/ankitjk/RunItBack), a separate project, classifies each moment of a match into
one of 11 scenes (`both_grounded`, `one_offstage`, `respawning` and so on;
definitions in [CLASSES.md](CLASSES.md)). Its bottleneck is labelled data:

- Only one video is labelled: 3 games, about 1,580 frames.
- On a game it has never seen, its scene model scores **48.3%** (measured in
  RunItBack, pooled over 3 leave-one-game-out folds).
- An earlier 92% came from a random frame split that put near-identical
  frames, 0.5 s apart, in both training and test. It was not a real score.

More labelled videos are the fix, and labelling by hand is slow: segments are
mostly 0.2 to 0.6 s long.

## How it works

```
MCP server (only door to data)        LangGraph workflow (typed state)

  predict_segments(video, game) <---  predict
  get_frame(video, t) -> JPEG   <--      |
  save_draft_labels(...)        <-    next_segment <----------------------+
     checks everything first             |                                |
                                         +- model is trusted -> accept ---+
                                         +- otherwise -------> llm_opinion|
                                         |                       +- agrees ----> accept --+
                                         |                       +- disagrees -> human ---+  (run pauses)
                                         |                       +- past the question cap -> defer
                                         +- no segments left -> save_draft -> END
```

One segment is handled per trip round the loop. With `--no-llm`, segments the
model is not trusted with go straight to `defer`. A finishing-blow zoom, or an
LLM answer that fails the checks, takes the same path as a disagreement.

| Actor | Job | Never does |
|---|---|---|
| Small model (ONNX, 8 classes) | Labels 2 frames per second, with a confidence. Neighbouring frames with the same label become one segment | Decide alone on `respawning` or `one_on_ledge`, which it predicts about 5 times too often |
| LLM (Gemini) | Looks at a segment's middle frame and says what each player is doing, with a one-line reason | Pick the class, decide where a segment goes, touch files |
| Person | Settles disagreements in the terminal: takes the model's label, the LLM's, or types another | - |
| Code | Turns the LLM's two player states into a class with a fixed ranking, routes every segment, checks everything before a save | - |

The coordinator is fixed code, not an LLM. The LLM is called only where
judgement is needed, and it is asked one narrow question.

**A segment is accepted without a person** when the model's lowest frame
confidence is at least 0.40 and its label is not `respawning` or
`one_on_ledge`, or when the LLM's class matches the model's. Anything else
goes to the person, up to 10 questions a run. Past that, the segment is
written with `needs_review: true`.

## Guardrails are in code, not in prompts

An instruction in a prompt is a request the LLM may not follow. A check in
code always runs.

- **The server is the only door to the data.** The workflow cannot write a
  file; it can only ask `save_draft_labels`, which refuses the whole save and
  names the segment and the rule if anything is wrong: an unknown class, a
  missing or unknown `source`, a segment outside the game or overlapping
  another, or a path outside `output/`.
- **A person's answer is never overwritten.** A later save cannot change or
  drop a segment with `source: human`.
- **The LLM's answer is checked before use** (`check_answer` in
  [labeler/llm.py](labeler/llm.py)): it must be well-formed, and each player
  state must be one of six. An answer that fails has no class and goes to the
  person. If the LLM names a class directly, it is ignored.
- **Every segment records who decided it**: `source` is `model`, `llm` or
  `human`, with the model's label, its confidence, and the LLM's label and
  reason kept beside the final label.
- **Hand labels are read-only.** Drafts go to `output/`, never over the
  ground-truth files. The prediction tool reads a game's start, end and stage
  from the hand-label file and never its labels.
- **The API key stays out of the repo and out of the server.** It is read from
  an environment variable by the workflow only; the server process is started
  without it.
- **The evaluation uses the graph's own routing functions**
  ([labeler/routing.py](labeler/routing.py)), so what is measured is what runs.

## Results

All results are from **game 3** of the labelled video, measured on 2026-11-24.
The small model was trained on games 1 and 2 only, so game 3 is the one game
it has never seen. The evaluation refuses to score any other game. Each draft
is compared with the hand labels every 0.1 s of game time (2,091 points).

**These results predate a label change.** On 2026-11-25 the GAME! screen at
the end of each game was relabelled from `non_gameplay` to `respawning`, to
match the written class definitions (4.05 s of game 3). Every number below was
measured against the labels before that change. The model-only numbers were
re-measured on 2026-11-26 against the new labels: accuracy 36.9% (was 35.7%),
segmentation ceiling 81.3% (81.3% before and after, to one decimal), 88.2% at
4 frames per second, and `respawning` precision 28.6% (was 20.6%). The LLM numbers have **not** been re-measured, so the tables below
keep the older, consistent set. The small model was trained before the change,
so it still learned GAME! as `non_gameplay`.

### The small model alone

| Measure | Value |
|---|---|
| Accuracy over all game time | 35.7% |
| Best any labels could score on the model's 218 segments ("segmentation ceiling") | 81.3% |
| Segments the model can be trusted with at threshold 0.40 | 11 of 218 (15.0% of game time, 5.4% of it wrong) |
| How often a `one_on_ledge` prediction is right | 9.0% |
| How often a `respawning` prediction is right | 20.6% |

The ceiling exists because a segment carries one label, and the model's
segments are cut on a 0.5 s grid while real scene changes fall anywhere.
Sampling 4 frames per second raises it to 88.2% (re-measured 2026-11-26 on the
new labels; the earlier 4 fps report read 88.3% and was overwritten), with
twice the segments.

### With the LLM's second opinion

Two Gemini models were shown the middle frame of all 218 segments.

| Measure | `gemini-4-hydrogen` | `gemini-3.5-flash-lite` |
|---|---|---|
| Accuracy if the LLM's label were used everywhere | **69.2%** | 50.5% |
| Answers that failed the code checks | 0 of 218 | 0 of 218 |
| Where model and LLM disagree: LLM right / model right / neither | 78.1 s / 6.1 s / 35.8 s | 56.6 s / 23.0 s / 53.8 s |
| Where they agree: share of that time that is wrong | 20.8% (of 84.1 s) | 22.6% (of 63.2 s) |
| Accepted without a person at threshold 0.40 | 61 of 218 segments | 46 of 218 |
| ...share of game time, and share of it wrong | 41.2%, 21.7% wrong | 32.4%, 22.2% wrong |
| Left for the person | 157 segments | 172 segments |

What this shows:

- **The LLM is far better than the small model** on an unseen game: 69.2%
  against 35.7%, with a ceiling of 81.3%.
- **"Both agree" is a weak reason to accept.** A fifth of the agreed time is
  wrong. Agreement on `one_on_ledge` is wrong 61.7% of the time, and on
  `respawning` 40.9% (`gemini-4-hydrogen`). The LLM's answer says more than
  its agreement with a weak model does, so the next version should route on
  the LLM's answer and stop requiring agreement.
- **Most segments still need a person**: 157 of 218, far more than a
  10-question cap. The rest of those are written as `needs_review`, carrying
  the LLM's label when it gave one and the model's otherwise.

### Read these numbers with care

- **35.7% is not comparable to the model's own test score of 39.2%.** That
  figure used 490 frames taken from inside each labelled segment, at least
  0.15 s from its edges. This evaluation covers all game time, including the
  moments where the scene changes.
- **The threshold was chosen on game 3 and is reported on game 3**, because
  it is the only unseen game. The 5.4% error at that threshold is therefore
  optimistic, rests on 11 segments, and needs a second unseen video to confirm.
- **Errors are counted by time**, so they include boundary error: 18.7% of
  game time cannot be right with these segments whatever label they get.
- **The LLM results are from before two changes**: one sentence added to the
  `non_gameplay` definition (the countdown before GO!), and the GAME! screen
  relabelled as `respawning`. They have not been re-measured with either.
- **The human step was run but not measured.** It pauses, takes answers,
  resumes and stops at the cap; how much a person's 10 answers improve a
  draft was not scored.
- **One video, one game, 209 seconds.** Nothing here says how the pipeline
  does on another stage, other characters or another video.

## Running it

It needs files that are not in this repo: the video, the hand labels (both in
a RunItBack checkout) and the trained model (`models/heldout_g3/`, or any
model folder named in `LABELER_MODEL_DIR`). Each draft records which model
and which LLM prompt produced it. The 0.40 confidence threshold was measured
for `heldout_g3` only: with any other model, nothing is accepted on the
model's word alone until a threshold is measured for it.

```
uv sync
uv run pytest                                  # 131 tests, no video or key needed
export GEMINI_API_KEY=...                      # from your keychain, not a file
uv run python -m labeler.run --game 3          # label a game; asks up to 10 questions
uv run python -m labeler.run --game 3 --no-llm --max-questions 0   # model only, nothing asked
uv run python -m labeler.evaluate --game 3 --llm                   # score against the hand labels
```

A question looks like this:

```
Question 1 of at most 10: 462.12-462.62 s (frame at 462.37 s: output/review/XUkRx6CpL04_462.37.jpg)
  1  one_on_ledge  (model, confidence 0.29)
  2  non_gameplay  (LLM: The match has not started yet, showing Wii Fit Trainer's pre-match entrance animation with the timer still at 7:00.00.)
  s  skip (leave it marked for review)   or type a class name
>
```

The draft is written to `output/<video>_g<game>_draft.json` in RunItBack's
label format, with the extra fields described in
[docs/DRAFT_FORMAT.md](docs/DRAFT_FORMAT.md). LLM answers are cached, so a
rerun asks nothing twice. The cache is keyed by the prompt, and the prompt is
built from `CLASSES.md`: after a change to the definitions, the next run asks
about every segment again, and the evaluation writes a new report beside the
old one (the report's file name carries an id for the prompt).
`gemini-4-hydrogen` needs a paid Google tier: the free tier allowed 20
requests a day when this was built (seen during the runs, not recorded in a
file), and one game needs about 200.

| Read this | For |
|---|---|
| [PLAN.md](PLAN.md) | Build steps, settled decisions, full results |
| [CLASSES.md](CLASSES.md) | The 11 classes and the ranking |
| [docs/DRAFT_FORMAT.md](docs/DRAFT_FORMAT.md) | The draft's fields and what the labelling tool can do with it |
| [docs/CONCEPTS.md](docs/CONCEPTS.md) | LangGraph, MCP and evaluation terms in plain words |

## Limits

- **It only runs on a video that already has a hand-label file**, because
  each game's start and end times are read from it. Running on a new video is
  not built.
- **RunItBack's labelling tool can open a draft but not correct it.** It
  cannot change or delete a loaded segment, and saving drops `source` and the
  other extra fields. Corrections happen in this pipeline's terminal review.
- **A second run on the same game is refused once a person has answered**,
  after its questions have been asked. The save check protects the earlier
  answers, and a new run does not yet read them back in.
- **The person sees one still frame per question**, in time order, and the
  paused run is held in memory: quitting loses the answers given so far.
- **The LLM sees one frame per segment**, so it cannot see a scene change
  inside a segment.
- **A zoomed-in finishing blow has no class yet.** The LLM flags these frames
  and they go to the person.

## Next

- **Route on the LLM's answer**, as the results suggest, and measure it on a
  second unseen video.
- **A context agent for disagreements** (designed, not built): an LLM that
  chooses which nearby frames to look at before answering. Code would limit
  it to one tool, 4 frames and 2 seconds either side of the segment, and
  accept its answer only when it matches the model or the first LLM.
- **Unlabelled videos**, which is the real use: treat a video with no label
  file as one game and let the person mark the game boundaries afterwards.
- **Load an existing draft**, so a run reuses answers already given.
- **Changes to RunItBack's labelling tool** so a person can correct a draft
  and keep its history (listed in [docs/DRAFT_FORMAT.md](docs/DRAFT_FORMAT.md)).
- **A full audit log**, beyond `source` per segment.
- **The retraining loop**: label a new video, retrain the small model, get
  better drafts.
