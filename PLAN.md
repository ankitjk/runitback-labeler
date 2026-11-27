# Build plan - first demo

Goal: a small working demo that pre-labels **game 3** of `XUkRx6CpL04`, routes
uncertain segments to a human, writes a draft label file the RunItBack GUI can
open, and **measures itself against the human labels**. Working and measured
beats feature-complete.

Keep the code small and readable enough that every part can be explained.
Plain-terms definitions of the LangGraph and MCP ideas used here are in
`docs/CONCEPTS.md`. The draft file format and how it meets the labelling GUI
are in `docs/DRAFT_FORMAT.md`.

## Before running (owner)

- **A Gemini API key** from Google AI Studio, exported as `GEMINI_API_KEY`
  from `~/.zprofile`, which reads it from the macOS Keychain. The key itself
  is never written to a file. Steps 1-4 run without one; the LLM steps need it.

## Architecture

```
MCP server (only door to data)        LangGraph workflow (typed state)

  predict_segments(video, game) <---  predict
  get_frame(video, t) -> JPEG   <--      |
  save_draft_labels(...)        <-    next_segment <---------------------------------+
     guardrails, in code                 |  (conditional edge: route)                |
                                         +- confident --> accept --------------------+
                                         +- unsure -----> llm_opinion                |
                                         |                  +- agrees ----> accept --+
                                         |                  +- disagrees -> context_agent
                                         |                  |                 +- sides with one -> accept
                                         |                  |                 +- can't tell, or
                                         |                  |                 |  a third answer --+
                                         |                  +- invalid, or zoom ------------------+-> human (interrupt)
                                         |                                                        |   or, past the question
                                         |                                                        |   cap, defer
                                         +- no segments left --> save_draft --> END
```

One segment is handled per trip round the loop. Most of the graph is fixed
code, and that is deliberate: LLMs are called only where judgement is needed,
and every decision about where a segment goes is made by code.

The context agent is the one agentic part: an LLM that chooses its own next
action (which frame to look at) in a loop. Everything around it is fixed.

### Who does what

| Actor | Job | Never does |
|---|---|---|
| Small model (ONNX) | First label for every frame, with a confidence | Decide alone on `respawning` or `one_on_ledge` |
| First-opinion LLM | Says what each player is doing in one frame, with a reason | Pick the final class (code does, from the ranking), choose where a segment goes, touch files |
| Context agent (LLM with a tool) | Settles a disagreement by choosing nearby frames to look at, then says what each player is doing | Call any tool but `get_frame`, look outside its time window, exceed its frame limit, pick the final class |
| Human | Settles disagreements; finishes the draft in the GUI | - |
| Code | Routing, the class ranking, all checks before saving | - |

## Decisions (settled; change them here first if they change)

1. **Python 3.12 via `uv`.** The system Python is 3.15.
2. **Pin `mcp>=1.30,<2`.** Version 2.x renamed FastMCP, and
   `langchain-mcp-adapters` 0.3.1 fails to import with it. Checked on
   2026-11-24: with `mcp` 1.30.0, `langgraph` 1.2.24, `onnxruntime` 1.32.0,
   `opencv-python-headless` 5.1.0 and `numpy` 2.5.5 on Python 3.12, all the
   imports work, the model loads and the video reads.
3. **Graph nodes call MCP tools with the plain `mcp` client**
   (`ClientSession.call_tool`), not `langchain-mcp-adapters`. The adapter exists
   to hand tools to an LLM that chooses which to call. Here, fixed code calls
   each tool at a fixed point, so the plain client is smaller and easier to
   explain. (This replaces the earlier plan to use the adapter.)
4. **Routing rules are pure functions** in `routing.py`, used by both the graph
   and the evaluation, so what is measured is exactly what runs.
5. **A segment goes to the first-opinion LLM if** its `min_confidence` is below the threshold,
   **or** the model's label is `respawning` or `one_on_ledge` (never accepted on
   the model's word alone: it predicts them 5x too often).
6. **The LLM reports each player's state, and code picks the class.** The LLM
   returns JSON: `gameplay` (true/false), `finishing_blow_zoom` (true/false),
   `state_a`, `state_b` (each one of respawning, offstage, on_ledge,
   knocked_down, airborne, grounded) and `reason`. `classes.py` applies the
   ranking from `CLASSES.md`. Anything that fails validation counts as a
   disagreement.
7. **Model and LLM agree -> accepted**, `source: model`, with the LLM's label
   and reason recorded next to it. This includes `respawning` and
   `one_on_ledge`: the evaluation reports their error rate separately so the
   rule can be tightened if agreement turns out not to be enough.
8. **Model and LLM disagree -> the context agent** (when
   `USE_CONTEXT_AGENT` is on). It gets the dispute and one tool, `get_frame`,
   and chooses which nearby frames to look at before answering. Its limits are
   enforced by the code that runs its tool calls, not by its prompt: at most
   `CONTEXT_MAX_FRAMES` (4) frames, only times within `CONTEXT_WINDOW_SEC`
   (2.0 s) of the segment and inside the game, and no other tool. A call that
   breaks a limit is not run; the agent gets an error message back instead.
   - Its class matches the model's or the first LLM's -> accepted,
     `source: llm`.
   - It says it cannot tell, gives a third class, flags a zoom, or gives an
     invalid answer -> the human.
9. **A segment goes to the human if** the first LLM's answer is invalid, either
   LLM flags a zoomed finishing blow (no class fits it yet, see `CLASSES.md`),
   or the context agent did not settle it. With `USE_CONTEXT_AGENT` off, every
   disagreement goes straight to the human. The switch exists so the agent can
   be turned off if its measured error rate is too high.
10. **Question cap.** After `MAX_QUESTIONS` human questions in a run, further
   segments bound for the human are *deferred*: written to the draft with
   `needs_review: true`. They take the LLM's label when it gave one
   (`source: llm`), otherwise the model's (`source: model`). This is the
   `DEFERRED_LABEL` setting, chosen from the evaluation: on disagreements the
   LLM is right far more often than the model (see Results). A question the
   person skips is deferred the same way. Deferred segments count as human
   work, never as auto-accepted.
11. **Bounds tolerance 0.05 s.** The ground-truth file's own first game-3 label
    starts at 437.1 s while the game starts at 437.1167 s, so a strict bounds
    check would reject real human labels.
12. **Draft path: `output/<video_id>_g<game>_draft.json`**, so a run on one game
    cannot replace another game's draft.
13. **Sampling: 2 frames per second** (config value). Each sampled frame stands
    for the 0.5 s around it. Human segments are often shorter than that, so
    some boundaries will be off. The evaluation measures how much this costs
    (the "segmentation ceiling" below).
14. **LLM provider: Google Gemini**, key in `GEMINI_API_KEY`. The first
    opinion runs on **`gemini-4-hydrogen`**, on a paid tier. Measured on
    2026-11-24: Google's free tier gives it 20 requests a day, and
    `gemini-3.5-flash` the same, while one game needs about 200.
    `gemini-3.5-flash-lite`, an older, lighter model, ran all 218 segments on
    the free tier and is the free fallback (`--llm-model`); it is clearly
    weaker (see Results). `gemini-3.8-flash` and `gemini-3.7-flash` answered
    only "busy" that evening. JSON output with a fixed answer shape works on
    every model that answered. Tool calls are still to be tested in step 7.
    Free-tier content is used by Google to improve its products. Answers are
    cached per model, so changing model needs no code change. The limits and
    "busy" replies above were observed during the runs, not recorded in a file.

## Layout

```
pyproject.toml
labeler/
  config.py      paths, sample rate, threshold, caps, LLM provider and model id
  classes.py     the 11 classes, the 6 player states, the ranking -> class   (pure)
  guardrails.py  validate_segments(): every check before a save            (pure)
  routing.py     after_predict(), after_llm(), after_context()             (pure)
  inference.py   preprocessing + ONNX + class_mapping.json   (server only)
  server.py      the MCP server: 3 tools
  mcp_client.py  opens the server, wraps the 3 tool calls
  llm.py         label_frame() -> the LLM's class and reason; check_answer(); answer cache
  context_agent.py  the tool-calling loop and its limits (step 7)
  graph.py       state, nodes, build_graph()
  run.py         CLI: runs the graph, asks the human, resumes
  evaluate.py    game-3 evaluation and threshold sweep
tests/           unit tests for the three pure modules
docs/            CONCEPTS.md, DRAFT_FORMAT.md
output/          drafts, review frames, caches, evaluation results (gitignored)
```

Only `server.py` and `inference.py` may open the video, the model or the
ground-truth file. The one exception is `evaluate.py`, which reads the human
labels for game 3 to score the draft; it is not part of the pipeline.

## Steps, in order (each one ends in something runnable)

Time estimates total about 10.5 hours. The order puts a measured result early. If
time runs out, see "Cut lines" below.

### 1. Setup (30 min)
- `uv init`, Python 3.12. Dependencies: `langgraph`, `mcp>=1.30,<2`,
  `onnxruntime`, `opencv-python-headless`, `numpy`, `pytest`. The LLM client is
  added in step 5.
- `config.py`: RunItBack root, video id, model directory, `SAMPLE_FPS = 2`,
  `CONFIDENCE_THRESHOLD` (placeholder until step 4 measures it),
  `NEVER_AUTO_ACCEPT = {"respawning", "one_on_ledge"}`, `MAX_QUESTIONS`,
  `BOUNDS_TOLERANCE_SEC = 0.05`.
- `classes.py`: the 11 classes, the 6 states, the ranking, and
  `scene_from_states(gameplay, state_a, state_b)`. Unit tests use the examples
  in `CLASSES.md` (one offstage + one on ledge -> `one_offstage`, and so on).

**Done when:** `uv run pytest` passes.

### 2. MCP server (2 h)
- `inference.py`: the four preprocessing lines from `CLAUDE.md`, an ONNX
  session, softmax, and the class order read from `class_mapping.json`.
- `predict_segments(video_id, game_number)`:
  - reads the game's `start_sec`/`end_sec` from the ground-truth file's `games`
    entry, and never its `labels`
  - does this lookup in one small function (`game_bounds()`), so other sources
    of game times can be added later without touching the rest
  - samples at `SAMPLE_FPS`, reading forward through the video (faster than
    seeking to each frame), and classifies in batches of 32
  - merges consecutive frames with the same class into segments; a boundary
    falls halfway between two sampled frames, and times are rounded to 0.01 s
  - each segment gets `class`, `start_sec`, `end_sec`, `mean_confidence`,
    `min_confidence`, `n_frames`
  - caches the result in `output/cache/`, keyed by video, game, sample rate and
    model file, so reruns and the sweep are instant
- `get_frame(video_id, t_sec, max_width=1280)`: one frame as a JPEG.
- `save_draft_labels(video_id, game_number, segments)`. All checks are in
  `guardrails.py` and run before anything is written:
  - every `class` is one of the 11 (so `finishing_blow` is refused)
  - every `source` is `model`, `llm` or `human`
  - each segment has `end_sec > start_sec` and lies within the game bounds
    (0.05 s tolerance)
  - segments are sorted and do not overlap
  - **human segments are kept:** if a draft already exists, every segment in it
    with `source: human` must appear unchanged in the new one
  - the output path must resolve to a file inside `output/`
  - on failure nothing is written, and the result lists **every** broken rule,
    naming the segment: `segment 12 (512.0-513.5): class 'finishing_blow' is
    not one of the 11 classes`
  - the file is written to a temporary name, then renamed, so a crash cannot
    leave half a file
- Input checks on all three tools: `video_id` must match
  `^[A-Za-z0-9_-]{11}$` and exist in the video cache (so a caller cannot pass a
  path), `game_number` must exist in the ground-truth file, and `t_sec` must be
  inside the video.
- Unit tests for `guardrails.py`: one test per rule, plus a valid case.
- Test the server on its own with the MCP inspector
  (`npx @modelcontextprotocol/inspector uv run python -m labeler.server`)
  before any LangGraph.

**Done when:** the inspector shows game-3 segments, a frame displays, a bad
save is refused with a clear message, and a good save writes a file that
`scene_labeler.py --labels` opens. Write down how many segments game 3
produces: it sets the LLM cost in step 5.

### 3. LangGraph graph, model only (1.5 h)
- State (a `TypedDict`): `video_id`, `game_number`, `segments`, `cursor` (index
  of the current segment), `questions_asked`, `save_result`. Each segment
  carries `start_sec`, `end_sec`, `model_label`, `mean_confidence`,
  `min_confidence`, and later `llm_label`, `llm_reason`, `context_label`,
  `context_reason`, `context_frames`, `final_label`, `source`, `needs_review`.
- `routing.py`: `after_predict(segment, threshold, never_auto_accept)` returns `"accept"` or
  `"llm"`. Unit-tested.
- Nodes: `predict` -> `next_segment` -> (`accept` | `defer`) -> back to
  `next_segment` -> `save_draft` when no segments are left. Until step 5 exists,
  the `"llm"` route goes to `defer`.
- Set `recursion_limit` explicitly when invoking the graph. LangGraph stops a
  run after a fixed number of steps to catch accidental infinite loops; this
  loop is intended and takes about 3 steps per segment.
- `mcp_client.py` starts the server as a subprocess (stdio) and wraps the three
  tool calls.

**Done when:** `uv run python -m labeler.run --game 3` writes
`output/XUkRx6CpL04_g3_draft.json` and the GUI opens it.

### 4. Evaluation, model only (1 h) **Do not skip.**
`evaluate.py` compares the model's segments (what `predict_segments` returns)
with the human labels for game 3. From step 5 it also scores the LLM's labels.

- **How time is compared:** at every 0.1 s of the game (the GUI rounds labels
  to 0.1 s). Each point takes the human label and the draft label covering it.
  Points no human label covers (there is a 0.1 s gap at 493.4-493.5 s) are
  left out.
- **Model-only accuracy**, overall and per class.
- **Segmentation ceiling:** the accuracy if every model segment were given its
  best possible single label. This is the most any second opinion can reach
  without changing segment boundaries.
- **Threshold sweep** (0.15-0.85 in steps of 0.05, plus "accept everything";
  on game 3 no segment's lowest frame confidence reaches 0.85; the highest
  is 0.80):
  for each threshold, how much game time is auto-accepted and how much of that
  is wrong. Uses the same `routing.py` functions as the graph.
- Results go to `output/eval/` as JSON plus a Markdown table.

Two honesty notes to carry into the README:
- This model-only accuracy will not equal the 39.2% in `test_metrics.json`.
  That figure used 490 frames taken from inside each labelled segment, 0.15 s
  away from its edges. This evaluation covers all game time, including the
  moments where the scene changes.
- The threshold is chosen on game 3 and reported on game 3, because game 3 is
  the only unseen game. The auto-accept error at the chosen threshold is
  therefore optimistic, and needs a second unseen video to confirm.

**Done when:** one command prints the table, and `CONFIDENCE_THRESHOLD` in
`config.py` is set from the sweep, with the reason noted beside it.

### 5. LLM second opinion (1.5 h)
- `llm.py`: `label_frame(video_id, t_sec, get_jpeg)` asks the LLM for
  `{gameplay, finishing_blow_zoom, state_a, state_b, reason}`, and
  `check_answer()` checks that answer in code and works out the class. The prompt includes the Terms and class definitions from
  `CLASSES.md`, read from the file at run time, not retyped. Provider and model
  id are config values; the key is read from an environment variable only.
- The `llm_opinion` node gets the segment's middle frame through `get_frame`,
  calls `label_frame`, validates the answer in code, and works out the class
  with `classes.py`.
- `routing.after_llm(segment, questions_asked, max_questions,
  use_context_agent)` returns `"accept"`, `"context"`, `"human"` or `"defer"`.
  Unit-tested, including an invalid answer and a zoom flag. Until step 7 is
  built, `USE_CONTEXT_AGENT` is off and `"context"` is never returned.
- Cache answers in `output/cache/`, keyed by video, time, model id and a prompt
  version, so reruns and the sweep cost nothing.
- Evaluation additions: get an LLM answer for **every** segment once (cached),
  then report LLM-only accuracy, model-plus-LLM coverage and auto-accept error
  per threshold, the error rate where both agree (overall, and for
  `respawning` and `one_on_ledge`), and who is right more often when they
  disagree (this sets `DEFERRED_LABEL`).

**Done when:** a run shows segments taking all three routes, and the
evaluation table includes the LLM rows.

### 6. Human-in-the-loop (1 h)
- The `human` node calls `interrupt()` with the times, both labels, both
  confidences and the LLM's reason. The run pauses; a checkpointer
  (`InMemorySaver`) holds the state.
- `run.py`: when the graph returns an interrupt, save the frame to
  `output/review/`, print the question, read the answer, and resume with
  `Command(resume=answer)`. `1` takes the model's label, `2` the LLM's, or type
  a class name. The answer is checked against the 11 classes in code; a typo
  asks again.
- The node sets `final_label`, `source: human`, and adds 1 to `questions_asked`.
- Keep the `human` node free of side effects before `interrupt()`: when a run
  resumes, LangGraph runs the node again from its first line.
- If time allows: swap `InMemorySaver` for `AsyncSqliteSaver`, so a run can be
  quit and picked up later at the same question.

**Done when:** a run pauses, takes an answer, resumes, stops asking at the
cap, and the draft holds `human` segments and `needs_review` segments.

### 7. Context agent (1.5 h) - only if steps 1-6 are done
- `context_agent.py`: a loop around an LLM that supports tool calls.
  1. Send the dispute: the segment's times, the model's label and confidence,
     the first LLM's label and reason, the labels of the two segments on each
     side, the segment's middle frame, and the definitions from `CLASSES.md`.
  2. The LLM either asks for a frame (`get_frame(t_sec)`) or gives its answer.
  3. For a frame request, code checks the limits from decision 8. If the call
     is allowed, code fetches the frame through the MCP client and sends it
     back; if not, it sends back an error message. Go to 2.
  4. The answer has the same fields as the first LLM's, plus `cannot_tell`.
     Code validates it and works out the class with `classes.py`.
  5. If the frame limit is reached with no answer, the result is `cannot_tell`.
- The `context_agent` node records `context_label`, `context_reason` and
  `context_frames` (the times it looked at) on the segment.
- `routing.after_context(segment, questions_asked, max_questions)` returns
  `"accept"`, `"human"` or `"defer"`. Unit-tested.
- Unit-test the limit check on its own (too many frames, a time outside the
  window, a time outside the game).
- Cache the agent's result per segment (video, times, model id, prompt
  version), including the frames it looked at.
- Check early that the chosen LLM client can receive an image as a tool
  result. If it cannot, send the frame as the next user message instead.
- Evaluation additions, at the chosen threshold: how many human questions the
  agent removes, the error rate of the labels it accepts, how often it sides
  with the model against the first LLM, and the average number of frames it
  looked at. If its accepted labels are wrong too often, leave
  `USE_CONTEXT_AGENT` off and report the numbers.

**Done when:** a run shows a disputed segment settled by the agent with the
frames it chose recorded in the draft, an out-of-window frame request is
refused, and the evaluation table has the with-agent and without-agent rows.

### 8. README (1 h)
For an outside reader: the problem (the data bottleneck and the 48.3% held-out
score), the diagram, who does what, the guardrails and why they are in code,
the measured game-3 results with the two honesty notes, the limits (below),
and next steps. Every number comes from a file in `output/eval/`.

## Results so far (game 3, measured 2026-11-24)

**Stale since 2026-11-25:** the GAME! screen was relabelled from `non_gameplay`
to `respawning` in the ground truth (4.05 s of game 3). Model-only re-measured
on 2026-11-26 against the new labels: 36.9% accuracy, ceiling 81.3% at 2 fps
and 88.2% at 4 fps; `respawning` precision 28.6% (was 20.6%); 37.5% at the
sampled frames (was 36.3%); at threshold 0.40 still 11 segments accepted, 5.4%
wrong.
The LLM numbers below are from the old labels and old prompt and are not
re-measured. The model was trained on the old labels.

Full tables are in `output/eval/` (regenerate with
`uv run python -m labeler.evaluate --game 3`, and `--fps 4` for the 4 fps
report).

- Model-only accuracy over all game time: **35.7%** (36.3% at the 419 sampled
  frames; `test_metrics.json` reports 39.2% on 490 frames taken from inside
  segments).
- Segmentation ceiling at 2 frames per second: **81.3%**. At 4 frames per
  second it is 88.2% (re-measured 2026-11-26; the earlier report read 88.3% and
  was overwritten), with 421 segments instead of 218 (twice the LLM calls).
  Sampling stays at 2 for now; revisit after step 5 shows the LLM's cost.
- Threshold **0.40**: the model alone is trusted for 11 of 218 segments, 15.0%
  of game time, 5.4% of it wrong. 25 of those 31 seconds are the opening
  non-gameplay screen. The other 207 segments need a second opinion. The
  threshold was chosen and measured on the same game, so 5.4% is optimistic,
  and it rests on 11 segments.
- The never-auto-accept rule holds up: `one_on_ledge` predictions are right
  9.0% of the time and `respawning` 20.6%.

### With the first-opinion LLM (measured 2026-11-24)

Two models were measured on the same 218 segments, with the class definitions
as they stood on 2026-11-24. One sentence was added to `non_gameplay` in
`CLASSES.md` on 2026-11-25 (the countdown before GO!); the prompt is built from
that file, so these numbers are from before that sentence and have not been
re-measured with it. Reports:
`output/eval/XUkRx6CpL04_g3_2fps_llm_<model>_v1_1ac4020f.md`. The last part
of the name identifies the prompt, so a run with changed definitions writes a
new report beside these and never replaces them
(`uv run python -m labeler.evaluate --game 3 --llm-model <model>`).

| Measure | `gemini-4-hydrogen` | `gemini-3.5-flash-lite` |
|---|---|---|
| LLM-only accuracy, all game time (model alone: 35.7%; ceiling: 81.3%) | **69.2%** | 50.5% |
| Answers that failed the code checks | 0 of 218 | 0 of 218 |
| Segments flagged as a finishing-blow zoom | 5 | 17 |
| Model and LLM agree: time, and share of it wrong | 84.1 s, 20.8% | 63.2 s, 22.6% |
| ...of which `one_on_ledge` | 6.0 s, 61.7% wrong | 6.0 s, 61.7% wrong |
| ...of which `respawning` | 11.0 s, 40.9% wrong | never agreed |
| They disagree: time; LLM right / model right / neither | 120.0 s; 78.1 / 6.1 / 35.8 s | 133.4 s; 56.6 / 23.0 / 53.8 s |
| At threshold 0.40: segments accepted without a human | 61 of 218 (11 model alone + 50 agreed) | 46 of 218 (11 + 35) |
| ...share of game time accepted, and share of it wrong | 41.2%, 21.7% | 32.4%, 22.2% |
| ...segments left for the human | 157 | 172 |

What this says:

- **The LLM is much better than the small model**, and `gemini-4-hydrogen` is
  much better than `gemini-3.5-flash-lite`. On disagreements hydrogen matches
  the human for 78.1 s against the model's 6.1 s, so `DEFERRED_LABEL` is `llm`.
- **Agreement is a weak reason to accept.** About a fifth of the agreed time
  is wrong with either LLM, and `one_on_ledge` and `respawning` are worse.
  Without the LLM, 15.0% of time was accepted with 5.4% wrong; with it, 41.2%
  with 21.7% wrong. Decision 7 (agree -> accept) should be revisited; nothing
  has been changed yet.
- Part of every error above is the segment boundaries, not the label: errors
  are counted by time, and 18.7% of game time cannot be right with these
  segments whatever label they get.
- Whether the zoom flags are right was not measured (the human labels have no
  zoom class).
- The hydrogen run made 213 calls and used 476,055 input tokens and 273,263
  output tokens (which includes the model's thinking). These counts are from
  the run's printed output, not from a saved file; 5 of the 218 answers were
  already cached, so a full game uses slightly more.

## Cut lines (if time runs short)

- After step 4: a working MCP server, a graph, and measured model-only results.
  This is the minimum demo.
- After step 5: the first LLM is added and measured. Describe the human step
  and the context agent as designed.
- After step 6: the full loop with a human. Describe the context agent as the
  next step.
- Always leave an hour for the README (step 8). A short README with real
  numbers matters more than step 7.

## Known limits (say them plainly)

- **The labelling GUI cannot correct a draft yet.** It loads a draft and shows
  it, but it cannot change or delete a loaded segment, and saving drops
  `source` and the other extra fields. The fix belongs in RunItBack
  (`docs/DRAFT_FORMAT.md` lists what is needed). Until then, human corrections
  happen in this pipeline's terminal review.
- **A second run on the same game is refused once a person has answered.**
  The save check will not let a later run change or drop a `source: human`
  segment, and a new run does not yet read the earlier answers back in. The
  refusal comes at the end, after the new run's questions have been asked, so
  those answers are lost. To run again, move the old draft aside first. Next step: a tool to load a
  draft, so a run reuses the answers already given and asks only new questions.
- **The human sees one question at a time in the terminal**, with the frame
  saved to `output/review/` to open. The paused state is held in memory, so
  quitting mid-run loses the answers given so far.
- One video, one unseen game. Every result is from game 3 alone.
- The first LLM sees one frame per segment. Only the context agent looks at
  neighbouring frames, and only for disputed segments, up to 4 frames.
- The two LLMs are coordinated by fixed code. They do not talk to each other.
- A segment's label is checked at its middle frame only; a segment that mixes
  two scenes gets one label.

## Out of scope for the first demo (these are "next")
- A full audit log (beyond `source` per segment)
- **Running on videos with no hand labels** (the real use):
  - if a video has no ground-truth file, treat the whole video as one game
    with stage `unknown`; the human marks the real game boundaries in the GUI
    afterwards (RunItBack's training split needs them)
  - an optional start and end time, to run on a short stretch and limit LLM cost
  - a demo run on a few minutes of a second video, with no accuracy claimed
  - a routine for hand-checking a sample of each new draft, since a new
    video's draft has no measured accuracy
- GUI changes in RunItBack so a human can correct a draft and keep its history
  (the four changes in `docs/DRAFT_FORMAT.md`)
- The retraining loop: label a new video -> retrain -> better pre-labels
- Making it general-purpose (pluggable classes, labellers and grouping key).
  Only `classes.py`, `CLASSES.md`, the model files and the "game" grouping are
  specific to Smash; keep it that way while building
- A web review UI, or community reviewers
