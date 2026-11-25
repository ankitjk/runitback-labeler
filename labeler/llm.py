"""The first-opinion LLM: one frame in, what each player is doing out.

The LLM never picks the scene class. It reports each player's state, and
`check_answer` (plain code) checks the answer and works out the class with the
ranking in classes.py. An answer that fails any check has no class.
"""

import asyncio
import hashlib
import json
import os
import re
from pathlib import Path

from labeler import config
from labeler.classes import PLAYER_STATES, scene_from_states

# Change this when the prompt or the answer format changes, so old cached answers are not reused.
PROMPT_VERSION = "v1"
MAX_REASON_CHARS = 300
RETRY_WAITS_SEC = (5, 15, 30, 60, 60)

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "gameplay": {"type": "boolean"},
        "finishing_blow_zoom": {"type": "boolean"},
        "state_a": {"type": "string", "enum": list(PLAYER_STATES), "nullable": True},
        "state_b": {"type": "string", "enum": list(PLAYER_STATES), "nullable": True},
        "reason": {"type": "string"},
    },
    "required": ["gameplay", "finishing_blow_zoom", "reason"],
}


def _section(markdown: str, heading: str) -> str:
    """The text under one '## heading' of a markdown file, up to the next '## '."""
    match = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", markdown, flags=re.M | re.S)
    if not match:
        raise ValueError(f"CLASSES.md has no section '## {heading}'")
    return match.group(1).strip()


def build_prompt(classes_md: str | None = None) -> str:
    """The instructions sent with every frame. The definitions are read from CLASSES.md, not retyped."""
    if classes_md is None:
        classes_md = (config.REPO_ROOT / "CLASSES.md").read_text()
    return "\n\n".join([
        "You are looking at one frame from a video of a two-player Super Smash Bros. Ultimate match. "
        "Report what the frame shows. Do not guess what happened before or after it.",
        "Answer with JSON:\n"
        "- gameplay: false if the frame is not gameplay (see non_gameplay below), otherwise true.\n"
        "- finishing_blow_zoom: true if the frame is the zoomed-in, slowed-down shot of a strong hit.\n"
        f"- state_a, state_b: what each of the two players is doing, each one of: {', '.join(PLAYER_STATES)}. "
        "Which player is A and which is B does not matter. Leave both out if gameplay is false.\n"
        "- reason: one short sentence saying what you see that decided it.",
        "If either player is respawning (see the respawning definition below), give that player the state respawning.",
        "The definitions below describe whole-frame classes. Use them to decide each player's state; "
        "do not answer with a class name.",
        "## Terms\n\n" + _section(classes_md, "Terms"),
        "## Classes\n\n" + _section(classes_md, "Classes"),
    ])


def check_answer(text: str) -> dict:
    """Check the LLM's raw answer in code. Returns llm_label (None if unusable), llm_zoom, llm_reason, llm_error."""
    def bad(error: str) -> dict:
        return {"llm_label": None, "llm_zoom": False, "llm_reason": "", "llm_error": error}

    try:
        answer = json.loads(text)
    except (TypeError, ValueError):
        return bad("the answer is not JSON")
    if not isinstance(answer, dict):
        return bad("the answer is not a JSON object")
    for field in ("gameplay", "finishing_blow_zoom"):
        if not isinstance(answer.get(field), bool):
            return bad(f"'{field}' is missing or not true/false")
    if not isinstance(answer.get("reason"), str):
        return bad("'reason' is missing or not text")
    reason = answer["reason"].strip()[:MAX_REASON_CHARS]

    if answer["finishing_blow_zoom"]:
        # No class fits a zoomed finishing blow yet (CLASSES.md), so there is no label to give.
        return {"llm_label": None, "llm_zoom": True, "llm_reason": reason, "llm_error": None}
    try:
        label = scene_from_states(answer["gameplay"], answer.get("state_a"), answer.get("state_b"))
    except ValueError as error:
        return bad(str(error))
    return {"llm_label": label, "llm_zoom": False, "llm_reason": reason, "llm_error": None}


class GeminiLabeler:
    """Asks Gemini about one frame. Answers are cached on disk, so a rerun asks nothing twice."""

    def __init__(self, model: str | None = None, cache_dir: Path | None = None):
        from google import genai  # imported here so the pure functions above need no SDK

        key = os.environ.get(config.LLM_API_KEY_ENV)
        if not key:
            raise RuntimeError(f"the environment variable {config.LLM_API_KEY_ENV} is not set")
        self.model = model or config.LLM_MODEL
        if not re.fullmatch(r"[A-Za-z0-9._-]+", self.model):
            # The id becomes part of the cache folder's name, so it must not be able to point elsewhere.
            raise ValueError(f"model id {self.model!r} may only contain letters, digits, '.', '_' and '-'")
        self.prompt = build_prompt()
        self._client = genai.Client(api_key=key)
        self._limit = asyncio.Semaphore(config.LLM_CONCURRENCY)
        prompt_id = hashlib.sha256(self.prompt.encode()).hexdigest()[:8]
        self._cache_dir = (cache_dir or config.OUTPUT_DIR / "cache" / "llm") / f"{self.model}_{PROMPT_VERSION}_{prompt_id}"
        self.calls_made = 0
        self.tokens_in = 0   # what the new calls cost: tokens sent...
        self.tokens_out = 0  # ...and tokens received, including the model's thinking

    def _cache_file(self, video_id: str, t_sec: float) -> Path:
        return self._cache_dir / f"{video_id}_{t_sec:.3f}.json"

    async def _ask(self, jpeg: bytes) -> str:
        from google.genai import errors, types

        request = dict(
            model=self.model,
            contents=[types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), self.prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_schema=ANSWER_SCHEMA, temperature=0
            ),
        )
        for wait in (*RETRY_WAITS_SEC, None):
            try:
                async with self._limit:
                    response = await self._client.aio.models.generate_content(**request)
                self.calls_made += 1
                usage = response.usage_metadata
                if usage is not None:
                    self.tokens_in += usage.prompt_token_count or 0
                    self.tokens_out += (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
                if not response.text:
                    # Not cached, so a later run asks again instead of counting this as a bad answer forever.
                    raise RuntimeError(f"{self.model} returned an empty answer")
                return response.text
            except errors.APIError as error:
                if error.code == 429 and "PerDay" in json.dumps(error.details or {}):
                    # Waiting a minute will not help. Answers so far are cached, so a later run carries on.
                    raise RuntimeError(f"the daily request limit for {self.model} is used up") from error
                # 429 = too many calls this minute, 5xx = the service is busy. Anything else will not fix itself.
                if wait is None or error.code not in (429, 500, 503):
                    raise
                await asyncio.sleep(wait)

    async def label_frame(self, video_id: str, t_sec: float, get_jpeg) -> dict:
        """`get_jpeg` is an async function that fetches the frame. It is called only if the answer is not cached."""
        cache_file = self._cache_file(video_id, t_sec)
        if cache_file.is_file():
            text = json.loads(cache_file.read_text())["text"]
        else:
            text = await self._ask(await get_jpeg())
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps({"model": self.model, "t_sec": t_sec, "text": text}))
        # The check runs on every use, cached or not.
        return check_answer(text)
