"""Stage 4 annotation Hydra entrypoint."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from vidaforge.common.hydra import (
    as_bool,
    object_dict,
    optional_int,
    optional_string,
    string_dict,
    string_tuple,
)
from vidaforge.annotation.camera import (
    CAMERA_PROMPT_VERSION,
    CameraConfig,
    CameraOrchestrator,
)
from vidaforge.annotation.caption import (
    CaptionConfig,
    CaptionOrchestrator,
)
from vidaforge.annotation.tag import (
    TagConfig,
    TagOrchestrator,
)
from vidaforge.serving.config import VLMInferenceConfig
from vidaforge.serving.vllm import VLLMServerPool, VLLMServerPoolConfig


def _build_vllm_server_pool_config(cfg: DictConfig) -> VLLMServerPoolConfig:
    serve = cfg.step.serve
    serve_name = str(serve.name)
    if serve_name != "vllm":
        raise ValueError(f"Unsupported Stage 4 serve backend: {serve_name!r}")
    return VLLMServerPoolConfig(
        model_path=str(serve.model_path),
        model_name=str(serve.model_name),
        replicas=serve.replicas,
        tp_size=int(serve.tp_size),
        cpu_per_replica=int(serve.ray_num_cpus),
        base_port=int(serve.base_port),
        host=str(serve.host),
        vllm_bin=str(serve.vllm_bin),
        api_key=str(serve.api_key),
        ray_address=str(cfg.ray_address),
        placement_strategy=str(serve.placement_strategy),
        health_poll_interval_sec=float(serve.health_poll_interval_sec),
        extra_args=string_tuple(serve.extra_args),
        env=string_dict(serve.env),
        log_dir=optional_string(serve.log_dir),
        allowed_local_media_path=optional_string(serve.allowed_local_media_path),
    )


def _build_camera_config(
    cfg: DictConfig,
    *,
    base_urls: tuple[str, ...],
    model: str,
    api_key: str,
) -> CameraConfig:
    output_path = Path(str(cfg.output_path))
    inference = _build_vlm_inference_config(
        cfg,
        base_urls=base_urls,
        model=model,
        api_key=api_key,
    )
    return CameraConfig(
        input_path=Path(str(cfg.input_path)),
        output_path=output_path,
        source=str(cfg.source),
        source_batch=str(cfg.source_batch),
        run_id=str(cfg.run_id),
        input_run_id=str(cfg.input_run_id),
        name=str(cfg.step.name),
        label_version=str(cfg.step.label_version),
        inference=inference,
        parquet_size=int(cfg.parquet_size),
        batch_size=int(cfg.step.client.batch_size),
        ray_num_cpus=float(cfg.step.client.ray_num_cpus),
        ray_address=str(cfg.ray_address),
        limit=optional_int(cfg.limit),
        resume=as_bool(cfg.step.resume),
    )


def _build_caption_config(
    cfg: DictConfig,
    *,
    base_urls: tuple[str, ...],
    model: str,
    api_key: str,
) -> CaptionConfig:
    inference = _build_vlm_inference_config(
        cfg,
        base_urls=base_urls,
        model=model,
        api_key=api_key,
    )
    return CaptionConfig(
        input_path=Path(str(cfg.input_path)),
        output_path=Path(str(cfg.output_path)),
        source=str(cfg.source),
        source_batch=str(cfg.source_batch),
        run_id=str(cfg.run_id),
        input_run_id=str(cfg.input_run_id),
        name=str(cfg.step.name),
        schema_version=str(cfg.step.schema_version),
        prompt_version=str(cfg.step.prompt_version),
        mode=str(cfg.step.mode),
        inference=inference,
        parquet_size=int(cfg.parquet_size),
        batch_size=int(cfg.step.client.batch_size),
        ray_num_cpus=float(cfg.step.client.ray_num_cpus),
        ray_address=str(cfg.ray_address),
        limit=optional_int(cfg.limit),
        resume=as_bool(cfg.step.resume),
    )


def _build_tag_config(
    cfg: DictConfig,
    *,
    base_urls: tuple[str, ...],
    model: str,
    api_key: str,
) -> TagConfig:
    inference = _build_vlm_inference_config(
        cfg,
        base_urls=base_urls,
        model=model,
        api_key=api_key,
    )
    return TagConfig(
        input_path=Path(str(cfg.input_path)),
        output_path=Path(str(cfg.output_path)),
        source=str(cfg.source),
        source_batch=str(cfg.source_batch),
        run_id=str(cfg.run_id),
        input_run_id=str(cfg.input_run_id),
        name=str(cfg.step.name),
        tag_schema_version=str(cfg.step.tag_schema_version),
        tag_prompt_version=str(cfg.step.tag_prompt_version),
        inference=inference,
        parquet_size=int(cfg.parquet_size),
        batch_size=int(cfg.step.client.batch_size),
        ray_num_cpus=float(cfg.step.client.ray_num_cpus),
        ray_address=str(cfg.ray_address),
        limit=optional_int(cfg.limit),
        resume=as_bool(cfg.step.resume),
    )


def _build_vlm_inference_config(
    cfg: DictConfig,
    *,
    base_urls: tuple[str, ...],
    model: str,
    api_key: str,
) -> VLMInferenceConfig:
    inference = cfg.step.inference
    return VLMInferenceConfig(
        base_urls=base_urls,
        api_key=api_key,
        model=model,
        media_input=str(inference.media_input),
        request_concurrency=int(inference.request_concurrency),
        trust_env=as_bool(inference.trust_env),
        temperature=float(inference.temperature),
        top_p=float(inference.top_p),
        presence_penalty=float(inference.presence_penalty),
        max_tokens=int(inference.max_tokens),
        extra_body=object_dict(inference.extra_body),
        store_prompt=as_bool(inference.store_prompt),
    )


def _run_camera_with_config(config: CameraConfig) -> None:
    pipeline = CameraOrchestrator(
        stage_name="stage4_annotation",
        step_name="step1_camera",
    )
    result = pipeline.build_camera(config)

    print(f"input_path={result.input_path}")
    print(f"source_count={result.source_count}")
    print(f"input_count={result.input_count}")
    print(f"resumed_count={result.resumed_count}")
    print(f"output_count={result.output_count}")
    print(f"ok_count={result.ok_count}")
    print(f"failed_count={result.failed_count}")
    print(f"shard_count={result.shard_count}")
    print(f"output_path={result.output_path}")
    print(f"summary_path={result.summary_path}")
    print(f"label_version={config.label_version}")
    print(f"prompt_version={CAMERA_PROMPT_VERSION}")
    print(f"base_urls={list(config.inference.base_urls)}")
    print(f"model={config.inference.model}")
    print(f"media_input={config.inference.media_input}")
    print(f"request_concurrency={config.inference.request_concurrency}")
    print(f"client_batch_size={config.batch_size}")
    print(f"client_ray_num_cpus={config.ray_num_cpus}")


def _run_camera(cfg: DictConfig) -> None:
    server_config = _build_vllm_server_pool_config(cfg)
    with VLLMServerPool(server_config) as server_pool:
        config = _build_camera_config(
            cfg,
            base_urls=server_pool.base_urls,
            model=server_pool.model,
            api_key=server_config.api_key,
        )
        _run_camera_with_config(config)


def _run_caption_with_config(config: CaptionConfig) -> None:
    pipeline = CaptionOrchestrator(
        stage_name="stage4_annotation",
        step_name="step2_caption",
    )
    result = pipeline.build_caption(config)

    print(f"input_path={result.input_path}")
    print(f"source_count={result.source_count}")
    print(f"input_count={result.input_count}")
    print(f"resumed_count={result.resumed_count}")
    print(f"output_count={result.output_count}")
    print(f"ok_count={result.ok_count}")
    print(f"failed_count={result.failed_count}")
    print(f"shard_count={result.shard_count}")
    print(f"output_path={result.output_path}")
    print(f"summary_path={result.summary_path}")
    print(f"schema_version={config.schema_version}")
    print(f"prompt_version={config.prompt_version}")
    print(f"mode={config.mode}")
    print(f"base_urls={list(config.inference.base_urls)}")
    print(f"model={config.inference.model}")
    print(f"media_input={config.inference.media_input}")
    print(f"request_concurrency={config.inference.request_concurrency}")
    print(f"client_batch_size={config.batch_size}")
    print(f"client_ray_num_cpus={config.ray_num_cpus}")


def _run_caption(cfg: DictConfig) -> None:
    server_config = _build_vllm_server_pool_config(cfg)
    with VLLMServerPool(server_config) as server_pool:
        config = _build_caption_config(
            cfg,
            base_urls=server_pool.base_urls,
            model=server_pool.model,
            api_key=server_config.api_key,
        )
        _run_caption_with_config(config)


def _run_tag_with_config(config: TagConfig) -> None:
    pipeline = TagOrchestrator(
        stage_name="stage4_annotation",
        step_name="step3_tag",
    )
    result = pipeline.build_tag(config)

    print(f"input_path={result.input_path}")
    print(f"source_count={result.source_count}")
    print(f"input_count={result.input_count}")
    print(f"resumed_count={result.resumed_count}")
    print(f"output_count={result.output_count}")
    print(f"ok_count={result.ok_count}")
    print(f"failed_count={result.failed_count}")
    print(f"shard_count={result.shard_count}")
    print(f"output_path={result.output_path}")
    print(f"summary_path={result.summary_path}")
    print(f"tag_schema_version={config.tag_schema_version}")
    print(f"tag_prompt_version={config.tag_prompt_version}")
    print(f"base_urls={list(config.inference.base_urls)}")
    print(f"model={config.inference.model}")
    print(f"media_input={config.inference.media_input}")
    print(f"request_concurrency={config.inference.request_concurrency}")
    print(f"client_batch_size={config.batch_size}")
    print(f"client_ray_num_cpus={config.ray_num_cpus}")


def _run_tag(cfg: DictConfig) -> None:
    server_config = _build_vllm_server_pool_config(cfg)
    with VLLMServerPool(server_config) as server_pool:
        config = _build_tag_config(
            cfg,
            base_urls=server_pool.base_urls,
            model=server_pool.model,
            api_key=server_config.api_key,
        )
        _run_tag_with_config(config)


@hydra.main(version_base=None, config_path="../configs/stage4_annotation", config_name="config")
def main(cfg: DictConfig) -> None:
    step_name = str(cfg.step.name)
    if step_name == "step1_camera":
        _run_camera(cfg)
        return
    if step_name == "step2_caption":
        _run_caption(cfg)
        return
    if step_name == "step3_tag":
        _run_tag(cfg)
        return
    raise ValueError(f"Unsupported Stage 4 annotation step: {step_name!r}")


if __name__ == "__main__":
    main()
