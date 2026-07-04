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

from .client import MediaInput, generate_camera_response
from .prompt import CameraPromptRequest, build_camera_prompt
from .parse import parse_camera_response
from .schema import (
    CAMERA_LABEL_VERSION,
    CAMERA_PROMPT_VERSION,
    unknown_camera_labels,
)


_OPENAI_REQUEST_TIMEOUT_SEC = 120


class CameraWorker:
    """Ray actor worker for Stage 4 camera QA rows."""

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
        store_prompt: bool = False,
        media_input: MediaInput = "local",
        temperature: float = 0.0,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        max_tokens: int = 2048,
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
            request: CameraPromptRequest | None = None
            camera_json: dict[str, object] | None = None
            camera_ok = 0
            camera_error = ""

            if int(row["frame_ok"]) != 1:
                camera_error = str(row["frame_error"])
            else:
                try:
                    frame_json = parse_json_object(
                        row["frame_json"],
                        description="frame_json",
                    )
                    request = build_camera_prompt(
                        frame_json=frame_json,
                        duration_sec=row["duration_sec"],
                    )
                    if not request.image_paths:
                        raise ValueError("camera prompt has no image_paths")

                    response_text = await generate_camera_response(
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
                    camera_json = parse_camera_response(
                        response_text,
                        model=self.model,
                    )
                    camera_ok = 1
                except Exception as exc:  # noqa: BLE001
                    camera_error = str(exc)
            if camera_json is None:
                camera_json = {
                    "label_version": CAMERA_LABEL_VERSION,
                    "prompt_version": CAMERA_PROMPT_VERSION,
                    **unknown_camera_labels(),
                    "model": "",
                    "ok": False,
                    "error": camera_error,
                }

            return self._build_row(
                row=row,
                camera_json=camera_json,
                camera_ok=camera_ok,
                camera_error=camera_error,
                request=request,
            )

    def _build_row(
        self,
        *,
        row: dict[str, object],
        camera_json: dict[str, object],
        camera_ok: int,
        camera_error: str,
        request: CameraPromptRequest | None = None,
    ) -> dict[str, object]:
        image_count = len(request.image_paths) if request is not None else 0
        timestamps_sec = list(request.timestamps_sec) if request is not None else []
        camera_row: dict[str, object] = dict(row)
        camera_row.update(
            {
                "camera_json": json.dumps(
                    camera_json,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "camera_ok": int(camera_ok),
                "camera_error": camera_error,
                "camera_prompt_image_count": image_count,
                "camera_prompt_timestamps_sec": json.dumps(
                    timestamps_sec,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "label_version": CAMERA_LABEL_VERSION,
                "prompt_version": CAMERA_PROMPT_VERSION,
                "source": str(row["source"]),
                "source_batch": str(row["source_batch"]),
                "input_run_id": self.input_run_id,
                "run_id": self.run_id,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            }
        )
        if request is not None and self.store_prompt:
            camera_row["camera_prompt_json"] = json.dumps(
                self._request_payload(request),
                ensure_ascii=False,
                sort_keys=True,
            )
        return camera_row

    @staticmethod
    def _request_payload(request: CameraPromptRequest) -> dict[str, object]:
        return {
            "label_version": request.label_version,
            "prompt_version": request.prompt_version,
            "system_prompt": request.system_prompt,
            "user_prompt": request.user_prompt,
            "image_paths": list(request.image_paths),
            "timestamps_sec": list(request.timestamps_sec),
        }
