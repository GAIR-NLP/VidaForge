from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import httpx
from openai import AsyncOpenAI

from vidaforge.common import (
    parse_json_object,
)

from .client import MediaInput, generate_caption_response
from .parse import parse_caption_response
from .prompt import CAPTION_PROMPT_VERSION, CaptionPromptRequest, build_caption_prompt
from .schema import CAPTION_SCHEMA_VERSION


_OPENAI_REQUEST_TIMEOUT_SEC = 120


class CaptionWorker:
    """Ray actor worker for Stage 4 caption rows."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        request_concurrency: int,
        trust_env: bool,
        run_id: str,
        input_run_id: str,
        mode: str = "video_audio",
        store_prompt: bool = False,
        media_input: MediaInput = "local",
        temperature: float = 0.0,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        max_tokens: int = 4096,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        if request_concurrency <= 0:
            raise ValueError("request_concurrency must be > 0")
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=httpx.AsyncClient(
                timeout=_OPENAI_REQUEST_TIMEOUT_SEC,
                trust_env=trust_env,
            ),
        )
        self.model = model
        self.run_id = run_id
        self.input_run_id = input_run_id
        self.mode = mode
        self.store_prompt = store_prompt
        self.media_input = media_input
        self.temperature = temperature
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.max_tokens = max_tokens
        self.extra_body = extra_body or {}
        self._semaphore = asyncio.Semaphore(request_concurrency)

    async def process_batch(
        self,
        rows: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        return await asyncio.gather(*(self._process_one(row) for row in rows))

    async def _process_one(self, row: dict[str, object]) -> dict[str, object]:
        async with self._semaphore:
            request: CaptionPromptRequest | None = None
            caption_json: dict[str, object] | None = None
            caption_ok = 0
            caption_error = ""

            if int(row["frame_ok"]) != 1:
                caption_error = str(row["frame_error"])
            else:
                try:
                    frame_json = parse_json_object(
                        row["frame_json"],
                        description="frame_json",
                    )
                    audio_json = parse_json_object(
                        row["audio_json"],
                        description="audio_json",
                        allow_empty_value=True,
                    )
                    camera_json = parse_json_object(
                        row["camera_json"],
                        description="camera_json",
                        allow_empty_value=True,
                    )
                    request = build_caption_prompt(
                        frame_json=frame_json,
                        audio_json=audio_json,
                        camera_json=camera_json,
                        duration_sec=row["duration_sec"],
                        mode=self.mode,  # type: ignore[arg-type]
                    )
                    if not request.image_paths:
                        raise ValueError("caption prompt has no image_paths")

                    response_text = await generate_caption_response(
                        client=self.client,
                        model=self.model,
                        request=request,
                        media_input=self.media_input,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        presence_penalty=self.presence_penalty,
                        max_tokens=self.max_tokens,
                        extra_body=self.extra_body,
                    )
                    caption_json = parse_caption_response(
                        response_text,
                        mode=self.mode,  # type: ignore[arg-type]
                        model=self.model,
                    )
                    caption_ok = 1
                except Exception as exc:  # noqa: BLE001
                    caption_error = str(exc)
            if caption_json is None:
                caption_json = {
                    "schema_version": CAPTION_SCHEMA_VERSION,
                    "prompt_version": CAPTION_PROMPT_VERSION,
                    "mode": self.mode,
                    "level_3": "",
                    "level_2": "",
                    "level_1": "",
                    "level_0": "",
                    "model": "",
                    "ok": False,
                    "error": caption_error,
                }

            return self._build_row(
                row=row,
                caption_json=caption_json,
                caption_ok=caption_ok,
                caption_error=caption_error,
                request=request,
            )

    def _build_row(
        self,
        *,
        row: dict[str, object],
        caption_json: dict[str, object],
        caption_ok: int,
        caption_error: str,
        request: CaptionPromptRequest | None = None,
    ) -> dict[str, object]:
        image_count = len(request.image_paths) if request is not None else 0
        timestamps_sec = list(request.timestamps_sec) if request is not None else []
        audio_paths = list(request.audio_paths) if request is not None else []
        caption_row: dict[str, object] = dict(row)
        caption_row.update(
            {
                "caption_json": json.dumps(
                    caption_json,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "caption_level_0": str(caption_json.get("level_0") or ""),
                "caption_level_1": str(caption_json.get("level_1") or ""),
                "caption_level_2": str(caption_json.get("level_2") or ""),
                "caption_level_3": str(caption_json.get("level_3") or ""),
                "caption_ok": int(caption_ok),
                "caption_error": caption_error,
                "caption_mode": str(
                    caption_json.get("mode") or (request.mode if request else "")
                ),
                "caption_prompt_image_count": image_count,
                "caption_prompt_timestamps_sec": json.dumps(
                    timestamps_sec,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "caption_prompt_audio_paths": json.dumps(
                    audio_paths,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "schema_version": CAPTION_SCHEMA_VERSION,
                "prompt_version": CAPTION_PROMPT_VERSION,
                "source": str(row["source"]),
                "source_batch": str(row["source_batch"]),
                "input_run_id": self.input_run_id,
                "run_id": self.run_id,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            }
        )
        if request is not None and self.store_prompt:
            caption_row["caption_prompt_json"] = json.dumps(
                self._request_payload(request),
                ensure_ascii=False,
                sort_keys=True,
            )
        return caption_row

    @staticmethod
    def _request_payload(request: CaptionPromptRequest) -> dict[str, object]:
        return {
            "prompt_version": request.prompt_version,
            "mode": request.mode,
            "system_prompt": request.system_prompt,
            "user_prompt": request.user_prompt,
            "image_paths": list(request.image_paths),
            "timestamps_sec": list(request.timestamps_sec),
            "audio_paths": list(request.audio_paths),
        }
