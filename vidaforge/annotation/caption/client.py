from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from vidaforge.serving.openai import (
    MediaInput,
    build_openai_multimodal_messages,
    generate_openai_structured_response,
)

from .prompt import CaptionPromptRequest
from .schema import CaptionStructuredOutput


def build_openai_caption_messages(
    request: CaptionPromptRequest,
    *,
    media_input: MediaInput = "local",
) -> list[dict[str, Any]]:
    """Build OpenAI-compatible multimodal chat messages for captioning."""
    return build_openai_multimodal_messages(
        system_prompt=request.system_prompt,
        user_prompt=request.user_prompt,
        image_paths=request.image_paths,
        audio_paths=request.audio_paths,
        media_input=media_input,
    )


async def generate_caption_response(
    *,
    client: AsyncOpenAI,
    model: str,
    request: CaptionPromptRequest,
    media_input: MediaInput = "local",
    temperature: float = 0.0,
    top_p: float = 1.0,
    presence_penalty: float = 0.0,
    max_tokens: int = 4096,
    extra_body: dict[str, Any] | None = None,
) -> str:
    """Call an OpenAI-compatible VLM endpoint and return raw caption JSON text."""
    return await generate_openai_structured_response(
        client=client,
        model=model,
        messages=build_openai_caption_messages(
            request,
            media_input=media_input,
        ),
        schema_name="caption_v1",
        schema=CaptionStructuredOutput.model_json_schema(),
        task_name="caption",
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
