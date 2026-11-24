# Concepts used in this project, in plain terms

Each idea, what it means, and where it shows up in this repo. File names refer
to the layout in `PLAN.md`; a file exists once its step is built.

## MCP (Model Context Protocol)

A standard way for a program to offer **tools** to an AI application. One side
is a server that says "here are the things I can do"; the other is a client
that calls them. Because it is a standard, any MCP client (this workflow, the
MCP inspector, a chat app) can use the same server unchanged.

| Term | Meaning | Here |
|---|---|---|
| **MCP server** | A program that offers tools | `labeler/server.py` |
| **Tool** | One named function with typed inputs and a described result | `predict_segments`, `get_frame`, `save_draft_labels` |
| **MCP client** | The program that calls the tools | `labeler/mcp_client.py`, used by the graph's nodes |
| **Transport** | How client and server talk | stdio: the client starts the server as a child process and they exchange messages over its input and output |
| **FastMCP** | The helper in the `mcp` Python package that turns a decorated function into a tool | `@mcp.tool()` in `server.py` |
| **MCP inspector** | A web page for calling a server's tools by hand | Used to test the server before the graph exists |

**Why the server is "the only door to the data".** The video, the model and
the label files are opened in one place only. So the rules about them live in
one place too, and no caller can skip them. The workflow cannot write a bad
label file, because it cannot write files at all: it can only ask
`save_draft_labels`, which checks everything first.

**Guardrail.** A rule that is enforced by code and cannot be talked round. An
instruction in an LLM prompt ("only use these 11 classes") is a request, and
the LLM may not follow it. A check in code (`if cls not in CLASSES: refuse`)
always runs. In this project every rule that matters is a check in code:
`labeler/guardrails.py` for saving, and the validation of the LLM's answer in
the `llm_opinion` node.

## LangGraph

A library for writing a workflow as a **graph**: boxes (steps) joined by
arrows (what runs next), with one shared record of data passed along.

| Term | Meaning | Here |
|---|---|---|
| **State** | The shared record every step reads and updates. Typed, so each field has a known shape | The `TypedDict` in `graph.py`: the video, the list of segments, which segment is current, how many questions were asked |
| **Node** | One step: a function that takes the state and returns the fields it changed | `predict`, `next_segment`, `accept`, `llm_opinion`, `context_agent`, `human`, `defer`, `save_draft` |
| **Edge** | A fixed arrow: after A, always run B | `accept` -> `next_segment` |
| **Conditional edge** | An arrow chosen by a function that looks at the state | After `next_segment`: accept, ask the LLM, or save. After `llm_opinion`: accept, send to the context agent, ask the human, or defer |
| **Routing** | The functions that choose a conditional edge | `labeler/routing.py` |
| **Interrupt** | A node pauses the whole run and hands a question to the outside. The run continues when an answer is sent back | The `human` node calls `interrupt()`; `run.py` shows the question and resumes with `Command(resume=answer)` |
| **Checkpointer** | Saves the state after every step. It is what makes pausing possible: the run can stop, and later continue from the saved state | `InMemorySaver` (or SQLite, to survive quitting the program) |
| **Thread id** | The name of one run, so the checkpointer knows which saved state to continue | Passed in the config when the graph is invoked |
| **Recursion limit** | The most steps one run may take before LangGraph stops it. It catches accidental endless loops | Raised on purpose: this graph loops once per segment |

## Agents

| Term | Meaning | Here |
|---|---|---|
| **LLM call** | One question, one answer. The LLM takes no actions | The first opinion: "what is each player doing in this frame?" |
| **Tool calling** | The LLM replies with a request to run a function ("get me the frame at 512.0 s"). Code runs it and sends the result back | The context agent asking for `get_frame` |
| **Agent** | An LLM in a loop that chooses its own next action, using tools, until it decides it is done | `labeler/context_agent.py` |
| **Agentic automation** | A workflow where some steps are done by agents, inside limits set by code | This whole pipeline |
| **Multi-agent system** | Several actors with different jobs, and a rule for who acts when | The first-opinion LLM and the context agent, coordinated by the fixed graph, with the small model and the human |
| **Coordinator** | Whatever decides which agent acts next. It can be an LLM or fixed code | Fixed code: the routing functions |

The context agent's freedom is narrow on purpose. It chooses *which frames to
look at*; it cannot choose which tools exist, how many frames it may see, how
far from the segment it may look, or what happens to its answer. Those limits
are checked by the code that runs its tool calls.

**Why a graph, and not one LLM agent deciding everything.** Here the path a
segment takes is decided by rules that can be read, tested and measured. The
LLM is asked one narrow question (what is each player doing in this frame?),
and code does the rest. The result is the same every time for the same
inputs, apart from the LLM's answer itself, and that answer is cached.

## The actors

- **The small model** runs on every sampled frame. It is fast and cheap, but
  weak on a game it has not seen.
- **The first-opinion LLM** gives a second opinion on segments the model is
  unsure about. It is slow and costs money, so it is used only where needed,
  and only while labelling. It reports what it sees in one frame; code turns
  that into a class using the ranking in `CLASSES.md`.
- **The context agent** is asked only when the model and the first LLM
  disagree. It looks at nearby frames of its choosing, then reports what it sees.
- **The human** settles disagreements and has the final say.
- **Code** decides who is asked, and checks everything before it is saved.

## Words used in the evaluation

| Term | Meaning |
|---|---|
| **Ground truth** | The human labels, taken as the right answers |
| **Held-out / unseen game** | A game the model was never trained on. Here, game 3 |
| **Confidence** | The model's own probability for the label it chose (0 to 1). A high value does not make it right, which is why the threshold is measured |
| **Threshold** | The confidence below which a segment gets a second opinion |
| **Auto-accepted** | Given its final label without a human: by the model alone, by the model and first LLM agreeing, or by the context agent settling a disagreement |
| **Coverage** | The share of game time that was auto-accepted |
| **Auto-accept error** | The share of auto-accepted time whose label differs from the human's. The most important number: these are mistakes nobody was asked about |
| **Human load** | How many segments a person had to decide |
| **Threshold sweep** | Running the evaluation at many thresholds to see the trade: a lower threshold accepts more, with more errors |
| **Source** | Per segment, who decided its label: `model`, `llm` or `human` |
