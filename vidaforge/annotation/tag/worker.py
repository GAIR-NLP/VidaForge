from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import httpx
from openai import AsyncOpenAI

from vidaforge.common import parse_json_object

from .client import MediaInput, generate_tag_response
from .parse import parse_tag_response
from .prompt import TagPromptRequest, build_tag_prompt
from .schema import (
    TAG_LABEL_FIELDS,
    TAG_PROMPT_VERSION,
    TAG_SCHEMA_VERSION,
    unknown_tag_labels,
)


_OPENAI_REQUEST_TIMEOUT_SEC = 120


class TagWorker:
    """Ray actor worker for Stage 4 semantic tag rows."""

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
            request: TagPromptRequest | None = None
            tag_json: dict[str, object] | None = None
            tag_ok = 0
            tag_error = ""

            if int(row["frame_ok"]) != 1:
                tag_error = str(row["frame_error"])
            else:
                try:
                    frame_json = parse_json_object(
                        row["frame_json"],
                        description="frame_json",
                    )
                    request = build_tag_prompt(
                        frame_json=frame_json,
                        duration_sec=row["duration_sec"],
                    )
                    if not request.image_paths:
                        raise ValueError("tag prompt has no image_paths")

                    response_text = await generate_tag_response(
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
                    tag_json = parse_tag_response(
                        response_text,
                        model=self.model,
                    )
                    tag_ok = 1
                except Exception as exc:  # noqa: BLE001
                    tag_error = str(exc)
            if tag_json is None:
                tag_json = {
                    "schema_version": TAG_SCHEMA_VERSION,
                    "prompt_version": TAG_PROMPT_VERSION,
                    **unknown_tag_labels(),
                    "model": "",
                    "ok": False,
                    "error": tag_error,
                }

            return self._build_row(
                row=row,
                tag_json=tag_json,
                tag_ok=tag_ok,
                tag_error=tag_error,
                request=request,
            )

    def _build_row(
        self,
        *,
        row: dict[str, object],
        tag_json: dict[str, object],
        tag_ok: int,
        tag_error: str,
        request: TagPromptRequest | None = None,
    ) -> dict[str, object]:
        image_count = len(request.image_paths) if request is not None else 0
        timestamps_sec = list(request.timestamps_sec) if request is not None else []
        tag_row: dict[str, object] = dict(row)
        tag_row.update(
            {
                "tag_json": json.dumps(tag_json, ensure_ascii=False, sort_keys=True),
                "tag_ok": int(tag_ok),
                "tag_error": tag_error,
                "tag_schema_version": TAG_SCHEMA_VERSION,
                "tag_prompt_version": TAG_PROMPT_VERSION,
                "tag_prompt_image_count": image_count,
                "tag_prompt_timestamps_sec": json.dumps(
                    timestamps_sec,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "source": str(row["source"]),
                "source_batch": str(row["source_batch"]),
                "input_run_id": self.input_run_id,
                "run_id": self.run_id,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            }
        )
        for field in TAG_LABEL_FIELDS:
            tag_row[f"tag_{field}"] = tag_json[field]
        if request is not None and self.store_prompt:
            tag_row["tag_prompt_json"] = json.dumps(
                self._request_payload(request),
                ensure_ascii=False,
                sort_keys=True,
            )
        return tag_row

    @staticmethod
    def _request_payload(request: TagPromptRequest) -> dict[str, object]:
        return {
            "schema_version": request.schema_version,
            "prompt_version": request.prompt_version,
            "system_prompt": request.system_prompt,
            "user_prompt": request.user_prompt,
            "image_paths": list(request.image_paths),
            "timestamps_sec": list(request.timestamps_sec),
        }
