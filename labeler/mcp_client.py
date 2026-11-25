"""Starts the MCP server as a child process and wraps its three tools."""

import base64
import json
import sys
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from labeler import config


class LabelerClient:
    """Use as:  async with LabelerClient() as client: ..."""

    async def __aenter__(self):
        self._stack = AsyncExitStack()
        # The server gets only these settings, not this process's whole environment,
        # so API keys held by the workflow never reach it.
        server = StdioServerParameters(
            command=sys.executable,
            args=["-m", "labeler.server"],
            cwd=str(config.REPO_ROOT),
            env={"RUNITBACK_ROOT": str(config.RUNITBACK_ROOT), "LABELER_OUTPUT_DIR": str(config.OUTPUT_DIR)},
        )
        read, write = await self._stack.enter_async_context(stdio_client(server))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        return self

    async def __aexit__(self, *exc_info):
        await self._stack.aclose()

    async def _call(self, tool: str, arguments: dict):
        result = await self._session.call_tool(tool, arguments)
        if result.isError:
            raise RuntimeError(f"{tool} failed: {result.content[0].text}")
        return result

    async def predict_segments(self, video_id: str, game_number: int, sample_fps: int | None = None) -> dict:
        arguments = {"video_id": video_id, "game_number": game_number}
        if sample_fps is not None:
            arguments["sample_fps"] = sample_fps
        result = await self._call("predict_segments", arguments)
        return json.loads(result.content[0].text)

    async def get_frame(self, video_id: str, t_sec: float, max_width: int = 1280) -> bytes:
        result = await self._call("get_frame", {"video_id": video_id, "t_sec": t_sec, "max_width": max_width})
        return base64.b64decode(result.content[0].data)

    async def save_draft_labels(self, video_id: str, game_number: int, segments: list[dict]) -> dict:
        result = await self._call(
            "save_draft_labels", {"video_id": video_id, "game_number": game_number, "segments": segments}
        )
        return json.loads(result.content[0].text)
