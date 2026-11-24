---
name: quiz
description: Quiz the owner on a part of this project so they can explain it aloud. Use when the owner types /quiz, or asks to be tested on, or to practise explaining, a step, file or concept (for example "/quiz mcp server" or "/quiz routing").
---

# Quiz

The owner must be able to explain every part of this project to someone else.
Ask questions about one part, listen to the answers, and correct them against
the real code.

## Choosing the topic

- If an argument was given (a step number, a file, or a concept such as
  "interrupts"), use it.
- Otherwise use the plan step most recently built in this session. If that is
  unclear, ask which part.

## Before asking

Read the code for that part, the matching section of `PLAN.md`, and the
matching entries in `docs/CONCEPTS.md`. If `CLAUDE.local.md` exists, read it
and favour the topics it lists. Ask only about things that exist in the repo
now, and take the right answers from the code, not from memory.

## How to run it

Ask **one question at a time** and wait for the answer. Ask five questions,
from these kinds, easiest first:

1. **What it does.** "What does `save_draft_labels` check before it writes?"
2. **Why it is built this way.** "Why are those checks in the tool and not in
   the LLM's prompt?"
3. **How data moves.** "The LLM disagrees with the model on a segment. Which
   state fields change, and which node runs next?"
4. **What happens when it goes wrong.** "The LLM returns a class that doesn't
   exist. What happens to that segment?"
5. **The hard follow-up.** "The model and the LLM agree and are both wrong.
   How would you know, and how often does it happen?" Answer with the measured
   number if one exists in `output/eval/`; if not, say it has not been measured.

After each answer:

- Say what was right.
- Give what was missing or wrong in one or two sentences, pointing to the code
  as `file:line`.
- If the answer was vague, give a model answer of two or three spoken
  sentences in plain words.

Then ask the next question. Do not give the answer before the owner has tried.

## At the end

List the two or three points to review, each with the file to reread. Offer
another round on the weakest one.

## Rules

- Plain words. Use a technical term only if `docs/CONCEPTS.md` defines it or
  you define it when you use it.
- Never invent a result. Numbers come from `output/eval/` or
  `models/heldout_g3/test_metrics.json`.
- Do not write quiz notes, or anything from `CLAUDE.local.md`, into a committed file.
