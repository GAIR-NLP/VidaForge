from __future__ import annotations

import json
import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.diffusion.train import TrainDiffusionRecipe, is_main_process


DEFAULT_CONFIG_PATH = "vidaforge_adapters/automodel/configs/wan2_1_t2v_flow.yaml"
SAMPLE_COUNT_FIELDS = (
    "video_latents",
    "image_latents",
    "latents",
    "text_embeddings",
    "text_embeddings_2",
)


def main(default_config_path: str = DEFAULT_CONFIG_PATH) -> None:
    cfg = parse_args_and_load_config(default_config_path)
    eval_options = _prepare_eval_config(cfg)

    recipe = TrainDiffusionRecipe(cfg)
    recipe.setup()
    _reset_eval_dataloader(recipe)
    recipe.model.eval()
    if hasattr(recipe.pipe, "eval"):
        recipe.pipe.eval()

    result = evaluate_loss(recipe, eval_options)
    if is_main_process():
        output_path = eval_options["output_path"]
        if output_path is not None:
            path = Path(output_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            logging.info("[EVAL] wrote %s", path)
        logging.info(
            "[EVAL] loss=%.8f samples=%s batches=%s cache_dir=%s checkpoint_dir=%s restore_from=%s",
            result["loss_mean"],
            result["sample_count"],
            result["batch_count"],
            result["cache_dir"],
            result["checkpoint_dir"],
            result["restore_from"],
        )


def evaluate_loss(recipe: TrainDiffusionRecipe, options: dict[str, Any]) -> dict[str, Any]:
    """Run a no-grad validation pass over the configured dataloader."""
    max_batches = options["max_batches"]
    log_every = options["log_every"]
    seed = options["seed"]
    check_loss = options["check_loss"]
    base_global_step = options["global_step"]

    collective_device = _collective_device(recipe)
    totals = torch.zeros(3, dtype=torch.float64, device=collective_device)
    local_loss_sum = totals[0]
    local_sample_count = totals[1]
    local_batch_count = totals[2]

    context_factory = getattr(recipe, "_transformer_engine_fp8_context", None)

    for batch_index, batch in enumerate(recipe.dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break

        batch_size = _count_batch_samples(batch)
        if batch_size <= 0:
            raise RuntimeError(f"eval batch has no countable samples at batch_index={batch_index}")

        _seed_eval_step(seed=seed, rank=_rank(), batch_index=batch_index)
        context = context_factory() if context_factory is not None else nullcontext()
        with torch.no_grad(), context:
            _, average_weighted_loss, _, _ = recipe.flow_matching_pipeline.step(
                model=recipe.model,
                batch=batch,
                device=recipe.device,
                dtype=recipe.compute_dtype,
                global_step=base_global_step + batch_index,
                collect_metrics=False,
                check_loss=check_loss,
            )

        loss = average_weighted_loss.detach().to(device=collective_device, dtype=torch.float64)
        local_loss_sum += loss * batch_size
        local_sample_count += batch_size
        local_batch_count += 1

        if log_every and log_every > 0 and (batch_index + 1) % log_every == 0 and is_main_process():
            logging.info("[EVAL] local_batch=%s local_loss=%.8f", batch_index + 1, float(loss.item()))

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)

    sample_count = int(totals[1].item())
    batch_count = int(totals[2].item())
    if sample_count <= 0:
        raise RuntimeError("eval dataloader produced no samples")

    loss_mean = float((totals[0] / totals[1]).item())
    sampler_summary = _sampler_summary(getattr(recipe, "sampler", None))

    return {
        "loss_mean": loss_mean,
        "sample_count": sample_count,
        "batch_count": batch_count,
        "local_batch_size": int(recipe.cfg.get("step_scheduler.local_batch_size")),
        "global_batch_size": int(recipe.cfg.get("step_scheduler.global_batch_size")),
        "dp_size": int(getattr(recipe, "dp_size", 1)),
        "world_size": int(getattr(recipe, "world_size", 1)),
        "max_batches": max_batches,
        "seed": seed,
        "shuffle": bool(recipe.cfg.get("data.dataloader.shuffle", False)),
        "drop_last": bool(recipe.cfg.get("data.dataloader.drop_last", True)),
        "cache_dir": str(recipe.cfg.get("data.dataloader.cache_dir")),
        "checkpoint_dir": str(recipe.cfg.get("checkpoint.checkpoint_dir")),
        "restore_from": recipe.cfg.get("checkpoint.restore_from", None),
        **sampler_summary,
    }


def _reset_eval_dataloader(recipe: TrainDiffusionRecipe) -> None:
    """Rebuild a fresh eval dataloader after checkpoint restore.

    AutoModel checkpoints include StatefulDataLoader state for training resume.
    Eval must ignore that cursor so every checkpoint is measured on the same
    validation set from the beginning.
    """
    dataloader_cfg = recipe.cfg.get("data.dataloader")
    if not hasattr(dataloader_cfg, "instantiate"):
        raise RuntimeError("data.dataloader must be a config node with instantiate()")

    dataloader, sampler = dataloader_cfg.instantiate(
        dp_rank=recipe._get_dp_rank(),
        dp_world_size=recipe._get_dp_group_size(),
        batch_size=recipe.cfg.get("step_scheduler.local_batch_size"),
    )
    object.__setattr__(recipe, "dataloader", dataloader)
    object.__setattr__(recipe, "sampler", sampler)

    step_scheduler = getattr(recipe, "step_scheduler", None)
    if step_scheduler is not None:
        step_scheduler.dataloader = dataloader


def _prepare_eval_config(cfg: Any) -> dict[str, Any]:
    """Apply eval-safe defaults while keeping AutoModel CLI overrides available."""
    checkpoint_dir = cfg.get("checkpoint.checkpoint_dir", None)
    require_checkpoint = bool(cfg.get("eval.require_checkpoint", True))
    restore_from = cfg.get("checkpoint.restore_from", None)
    checkpoint_enabled = bool(cfg.get("checkpoint.enabled", True))
    if not checkpoint_enabled and (require_checkpoint or restore_from is not None):
        raise ValueError(
            "checkpoint.enabled=false is only valid for fresh-init eval. "
            "Set --eval.require_checkpoint false and leave --checkpoint.restore_from unset."
        )
    if require_checkpoint and restore_from is None and not _has_checkpoint_subdir(checkpoint_dir):
        raise ValueError(
            "eval requires a checkpoint. Set --checkpoint.restore_from <ckpt_dir_or_name>, "
            "or set --eval.require_checkpoint false to evaluate the base pretrained model."
        )

    if not bool(cfg.get("eval.enable_wandb", False)):
        cfg.__dict__.pop("wandb", None)

    cfg.set_by_dotted("data.dataloader.shuffle", bool(cfg.get("eval.shuffle", False)))
    cfg.set_by_dotted("data.dataloader.drop_last", bool(cfg.get("eval.drop_last", True)))

    if cfg.get("eval.num_workers", None) is not None:
        cfg.set_by_dotted("data.dataloader.num_workers", int(cfg.get("eval.num_workers")))
    if cfg.get("eval.prefetch_factor", None) is not None:
        cfg.set_by_dotted("data.dataloader.prefetch_factor", int(cfg.get("eval.prefetch_factor")))
    if cfg.get("eval.limit", None) is not None:
        cfg.set_by_dotted("data.dataloader.limit", int(cfg.get("eval.limit")))

    output_path = cfg.get("eval.output_path", None)
    if output_path is None and checkpoint_dir:
        restore_label = Path(str(restore_from)).name if restore_from else "latest"
        output_path = str(Path(str(checkpoint_dir)).expanduser() / f"eval_loss_{restore_label}.json")

    max_batches = cfg.get("eval.max_batches", None)
    if max_batches is not None:
        max_batches = int(max_batches)
        if max_batches <= 0:
            raise ValueError("eval.max_batches must be > 0 when set")

    log_every = int(cfg.get("eval.log_every", 20))
    if log_every < 0:
        raise ValueError("eval.log_every must be >= 0")

    return {
        "max_batches": max_batches,
        "log_every": log_every,
        "seed": int(cfg.get("eval.seed", 12345)),
        "check_loss": bool(cfg.get("eval.check_loss", False)),
        "global_step": int(cfg.get("eval.global_step", 0)),
        "output_path": output_path,
    }


def _count_batch_samples(batch: dict[str, Any]) -> int:
    for field in SAMPLE_COUNT_FIELDS:
        value = batch.get(field)
        if value is not None and hasattr(value, "shape") and len(value.shape) > 0:
            return int(value.shape[0])
    return 0


def _seed_eval_step(*, seed: int, rank: int, batch_index: int) -> None:
    step_seed = int(seed) + int(rank) * 1_000_003 + int(batch_index)
    torch.manual_seed(step_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(step_seed)


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return 0


def _collective_device(recipe: TrainDiffusionRecipe) -> torch.device:
    if dist.is_available() and dist.is_initialized() and str(dist.get_backend()).lower() == "nccl":
        return recipe.device
    return torch.device("cpu")


def _has_checkpoint_subdir(checkpoint_dir: object) -> bool:
    if checkpoint_dir is None:
        return False
    path = Path(str(checkpoint_dir)).expanduser()
    if not path.is_dir():
        return False
    return any(child.is_dir() and child.name.startswith("epoch_") for child in path.iterdir())


def _sampler_summary(sampler: object) -> dict[str, int]:
    if sampler is None:
        return {}

    result: dict[str, int] = {}
    bucket_groups = getattr(getattr(sampler, "dataset", None), "bucket_groups", None)
    if isinstance(bucket_groups, dict):
        result["bucket_count"] = len(bucket_groups)
        result["metadata_item_count"] = sum(
            len(group.get("indices", [])) for group in bucket_groups.values() if isinstance(group, dict)
        )

    usable_fn = getattr(sampler, "bucket_usable_counts", None)
    if callable(usable_fn):
        usable_counts = usable_fn()
        result["usable_sample_count"] = int(sum(int(value) for value in usable_counts.values()))

    dropped_fn = getattr(sampler, "bucket_dropped_counts", None)
    if callable(dropped_fn):
        dropped_counts = dropped_fn()
        result["dropped_sample_count"] = int(sum(int(value) for value in dropped_counts.values()))

    return result


if __name__ == "__main__":
    main()
