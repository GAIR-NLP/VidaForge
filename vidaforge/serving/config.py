from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


MediaInputConfig = Literal["local", "base64"]


@dataclass(slots=True)
class VLMInferenceConfig:
    """Client-side config for OpenAI-compatible multimodal VLM inference."""

    base_urls: tuple[str, ...] = ()
    api_key: str = "EMPTY"
    model: str = ""
    media_input: MediaInputConfig = "local"
    request_concurrency: int = 1
    trust_env: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    presence_penalty: float = 0.0
    max_tokens: int = 2048
    extra_body: dict[str, Any] | None = None
    store_prompt: bool = False


def validate_vlm_inference_config(config: VLMInferenceConfig) -> None:
    if not config.base_urls:
        raise ValueError("inference.base_urls must be non-empty")
    for base_url in config.base_urls:
        if not str(base_url).strip():
            raise ValueError("inference.base_urls must not contain empty values")
    if not str(config.model).strip():
        raise ValueError("inference.model must be set")
    if config.media_input not in {"local", "base64"}:
        raise ValueError("inference.media_input must be one of: local, base64")
    if config.request_concurrency <= 0:
        raise ValueError("inference.request_concurrency must be > 0")
    if not 0 <= config.top_p <= 1:
        raise ValueError("inference.top_p must be between 0 and 1")
    if config.max_tokens <= 0:
        raise ValueError("inference.max_tokens must be > 0")


def vlm_inference_summary(config: VLMInferenceConfig) -> dict[str, object]:
    return {
        "base_url": config.base_urls[0] if config.base_urls else "",
        "base_urls": list(config.base_urls),
        "model": config.model,
        "media_input": config.media_input,
        "request_concurrency": config.request_concurrency,
        "trust_env": config.trust_env,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "presence_penalty": config.presence_penalty,
        "max_tokens": config.max_tokens,
        "extra_body": config.extra_body or {},
        "store_prompt": config.store_prompt,
    }
