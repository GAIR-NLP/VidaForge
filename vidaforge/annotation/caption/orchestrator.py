from __future__ import annotations

import time

from vidaforge.common import utc_now_iso, write_summary_json
from vidaforge.index import (
    count_parquet,
    run_ray_async_actor_processing,
)
from vidaforge.serving.config import (
    validate_vlm_inference_config,
    vlm_inference_summary,
)

from .config import CaptionConfig, CaptionResult
from .prompt import CAPTION_PROMPT_VERSION
from .schema import CAPTION_SCHEMA_VERSION
from .worker import CaptionWorker


def _validate_config(config: CaptionConfig) -> None:
    for field in ("source", "source_batch", "run_id", "input_run_id"):
        if not str(getattr(config, field)).strip():
            raise ValueError(f"{field} is required")
    if config.schema_version != CAPTION_SCHEMA_VERSION:
        raise ValueError(f"Unsupported caption schema_version: {config.schema_version!r}")
    if config.prompt_version != CAPTION_PROMPT_VERSION:
        raise ValueError(f"Unsupported caption prompt_version: {config.prompt_version!r}")
    if config.mode not in {"video", "video_audio"}:
        raise ValueError("mode must be one of: video, video_audio")
    validate_vlm_inference_config(config.inference)
    if config.parquet_size <= 0:
        raise ValueError("parquet_size must be > 0")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if config.ray_num_cpus <= 0:
        raise ValueError("ray_num_cpus must be > 0")


class CaptionOrchestrator:
    """Orchestrate Stage 4 Step 2 captioning over extracted context rows."""

    def __init__(
        self,
        stage_name: str = "stage4_annotation",
        step_name: str = "step2_caption",
    ) -> None:
        self.stage_name = stage_name
        self.step_name = step_name

    def build_caption(
        self,
        config: CaptionConfig,
    ) -> CaptionResult:
        _validate_config(config)

        input_path = config.input_path.expanduser().resolve()
        output_path = config.output_path.expanduser().resolve()
        source_count = count_parquet(input_path, unit="clip")

        inference = config.inference
        started_at = utc_now_iso()
        started_perf = time.perf_counter()
        output_path.mkdir(parents=True, exist_ok=True)

        runtime_stats, writer_summary = run_ray_async_actor_processing(
            input_path=input_path,
            output_path=output_path,
            parquet_size=config.parquet_size,
            input_unit="clip",
            output_unit="clip",
            step="caption",
            ray_address=config.ray_address,
            actor_cls=CaptionWorker,
            actor_options={
                "num_cpus": config.ray_num_cpus,
                "num_gpus": 0,
            },
            actor_kwargs=[
                {
                    "base_url": base_url,
                    "api_key": inference.api_key,
                    "model": inference.model,
                    "request_concurrency": inference.request_concurrency,
                    "trust_env": inference.trust_env,
                    "run_id": config.run_id,
                    "input_run_id": config.input_run_id,
                    "mode": config.mode,
                    "store_prompt": inference.store_prompt,
                    "media_input": inference.media_input,
                    "temperature": inference.temperature,
                    "top_p": inference.top_p,
                    "presence_penalty": inference.presence_penalty,
                    "max_tokens": inference.max_tokens,
                    "extra_body": inference.extra_body,
                }
                for base_url in inference.base_urls
            ],
            batch_size=config.batch_size,
            limit=config.limit,
            resume=config.resume,
            desc="caption",
        )
        elapsed_sec = round(time.perf_counter() - started_perf, 3)

        summary = {
            "created_at": utc_now_iso(),
            **writer_summary,
            "stage": self.stage_name,
            "step": config.name,
            "schema_version": config.schema_version,
            "prompt_version": config.prompt_version,
            "mode": config.mode,
            **vlm_inference_summary(config.inference),
            "resume": config.resume,
            "parquet_size": config.parquet_size,
            "batch_size": config.batch_size,
            "ray_num_cpus": config.ray_num_cpus,
            "source_count": source_count,
            "input_count": runtime_stats.input_count,
            "resumed_count": runtime_stats.resumed_count,
            "output_count": runtime_stats.output_count,
            "ok_count": runtime_stats.ok_count,
            "failed_count": runtime_stats.failed_count,
            "ok_count_with_resume": runtime_stats.ok_count + runtime_stats.resumed_count,
            "started_at": started_at,
            "finished_at": utc_now_iso(),
            "elapsed_sec": elapsed_sec,
            "failed_examples": runtime_stats.failed_examples,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "run_id": config.run_id,
            "input_run_id": config.input_run_id,
            "source": config.source,
            "source_batch": config.source_batch,
            "limit": config.limit,
        }
        summary_path = write_summary_json(summary, output_path)

        return CaptionResult(
            input_path=input_path,
            output_path=output_path,
            source_count=source_count,
            input_count=runtime_stats.input_count,
            resumed_count=runtime_stats.resumed_count,
            output_count=runtime_stats.output_count,
            ok_count=runtime_stats.ok_count,
            failed_count=runtime_stats.failed_count,
            shard_count=int(writer_summary["shard_count"]),
            summary_path=summary_path,
            elapsed_sec=elapsed_sec,
        )
