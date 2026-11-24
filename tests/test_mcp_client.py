"""The real client talking to the real server, over a tiny made-up video in a temporary folder."""

import asyncio
import json

import cv2
import numpy as np
import pytest

from labeler import config
from labeler.mcp_client import LabelerClient

VIDEO_ID = "TESTvideo01"


@pytest.fixture
def world(tmp_path, monkeypatch):
    root, output = tmp_path / "runitback", tmp_path / "output"
    videos, truth = root / ".video_cache", root / "data" / "scene_classifier" / "raw_labels"
    videos.mkdir(parents=True)
    truth.mkdir(parents=True)
    monkeypatch.setattr(config, "RUNITBACK_ROOT", root)
    monkeypatch.setattr(config, "OUTPUT_DIR", output)

    writer = cv2.VideoWriter(str(videos / f"{VIDEO_ID}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 180))
    for i in range(50):
        writer.write(np.full((180, 320, 3), i * 5, dtype=np.uint8))
    writer.release()
    (truth / f"{VIDEO_ID}.json").write_text(json.dumps({
        "video_id": VIDEO_ID,
        "games": [{"game_number": 1, "stage": "battlefield", "start_sec": 1.0, "end_sec": 4.0, "labels": []}],
    }))
    return output


def test_client_and_server_end_to_end(world):
    async def scenario():
        async with LabelerClient() as client:
            jpeg = await client.get_frame(VIDEO_ID, 2.0, max_width=160)
            refused = await client.save_draft_labels(
                VIDEO_ID, 1, [{"start_sec": 1.0, "end_sec": 2.0, "class": "finishing_blow", "source": "model"}])
            saved = await client.save_draft_labels(
                VIDEO_ID, 1, [{"start_sec": 1.0, "end_sec": 4.0, "class": "both_grounded", "source": "model"}])
            with pytest.raises(RuntimeError, match="not a valid id"):
                await client.get_frame("../../secret", 1.0)
        return jpeg, refused, saved

    jpeg, refused, saved = asyncio.run(scenario())
    assert jpeg[:2] == b"\xff\xd8"
    assert refused["saved"] is False and "finishing_blow" in refused["errors"][0]
    assert saved["saved"] is True
    # The server used the temporary folders it was given, not the real ones.
    assert (world / f"{VIDEO_ID}_g1_draft.json").is_file()
