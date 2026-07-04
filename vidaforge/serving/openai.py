from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Literal, Sequence

from openai import AsyncOpenAI


MediaInput = Literal["local", "base64"]
MediaKind = Literal["image", "audio"]


_MEDIA_TYPES: dict[MediaKind, dict[str, str]] = {
    "image": {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    },
    "audio": {
        ".m4a": "audio/mp4",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
    },
}


def media_data_url(path: str | Path, *, kind: MediaKind) -> str:
    media_path = Path(path).expanduser().resolve()
    media_type = _MEDIA_TYPES[kind].get(media_path.suffix.lower())
    if media_type is None:
        raise ValueError(f"unsupported {kind} media type: {media_path.suffix!r}")

    encoded = base64.b64encode(media_path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def media_url(path: str | Path, *, media_input: MediaInput, kind: MediaKind) -> str:
    if media_input == "local":
        return f"file://{Path(path).expanduser().resolve()}"
    if media_input == "base64":
        return media_data_url(path, kind=kind)
    raise ValueError(f"unsupported media_input: {media_input!r}")


def build_openai_multimodal_messages(
    *,
    system_prompt: str,
    user_prompt: str,
    image_paths: Sequence[str | Path] = (),
    audio_paths: Sequence[str | Path] = (),
    media_input: MediaInput = "local",
) -> list[dict[str, Any]]:
    """Build OpenAI-compatible chat messages with text, images, and optional audio."""
    user_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": user_prompt,
        }
    ]
    for image_path in image_paths:
        image_url = media_url(image_path, media_input=media_input, kind="image")
        user_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
                "uuid": image_url,
            }
        )
    for audio_path in audio_paths:
        audio_url = media_url(audio_path, media_input=media_input, kind="audio")
        user_content.append(
            {
                "type": "audio_url",
                "audio_url": {
                    "url": audio_url,
                },
                "uuid": audio_url,
            }
        )
    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


async def generate_openai_structured_response(
    *,
    client: AsyncOpenAI,
    model: str,
    messages: list[dict[str, Any]],
    schema_name: str,
    schema: dict[str, Any],
    task_name: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
    presence_penalty: float = 0.0,
    max_tokens: int = 2048,
    extra_body: dict[str, Any] | None = None,
) -> str:
    """Call an OpenAI-compatible endpoint with strict JSON schema output."""
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        },
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    if not response.choices:
        raise ValueError(f"{task_name} model returned no choices")
    content = response.choices[0].message.content
    if not content:
        raise ValueError(f"{task_name} model returned empty response content")
    return content
