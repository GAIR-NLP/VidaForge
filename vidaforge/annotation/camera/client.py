from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from vidaforge.serving.openai import (
    MediaInput,
    build_openai_multimodal_messages,
    generate_openai_structured_response,
)

from .prompt import CameraPromptRequest
from .schema import camera_structured_output_json_schema


def build_openai_camera_messages(
    request: CameraPromptRequest,
    *,
    media_input: MediaInput = "local",
) -> list[dict[str, Any]]:
    """Build OpenAI-compatible multimodal chat messages for camera QA."""
    return build_openai_multimodal_messages(
        system_prompt=request.system_prompt,
        user_prompt=request.user_prompt,
        image_paths=request.image_paths,
        media_input=media_input,
    )


async def generate_camera_response(
    *,
    client: AsyncOpenAI,
    model: str,
    request: CameraPromptRequest,
    media_input: MediaInput = "local",
    temperature: float = 0.0,
    top_p: float = 1.0,
    presence_penalty: float = 0.0,
    max_tokens: int = 2048,
    extra_body: dict[str, Any] | None = None,
) -> str:
    """Call an OpenAI-compatible VLM endpoint and return raw response text."""
    return await generate_openai_structured_response(
        client=client,
        model=model,
        messages=build_openai_camera_messages(
            request,
            media_input=media_input,
        ),
        schema_name="camera_v1",
        schema=camera_structured_output_json_schema(),
        task_name="camera",
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
