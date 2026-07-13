from __future__ import annotations

import argparse
import copy
import json
import os
import pprint
import random
import sys
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate V-JEPA2 checkpoint loss on a V-JEPA manifest."
    )
    parser.add_argument("--fname", required=True, help="V-JEPA2 training YAML config.")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path to evaluate.")
    parser.add_argument("--valid-csv", required=True, help="Validation manifest CSV.")
    parser.add_argument("--output", required=True, help="Rank-0 JSON output path.")
    parser.add_argument(
        "--vjepa2-dir",
        default=None,
        help="Optional path to the V-JEPA2 repository. Prepended to sys.path before imports.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional local batches per rank for smoke testing. Omit for full validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override meta.seed for deterministic eval sampling.",
    )
    parser.add_argument(
        "--lambda-mode",
        choices=("checkpoint", "fixed"),
        default="checkpoint",
        help=(
            "checkpoint uses the training lambda schedule at the checkpoint epoch; "
            "fixed always uses model.lambda_value_vid."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("--max-batches must be > 0 when set")

    if args.vjepa2_dir:
        sys.path.insert(0, str(Path(args.vjepa2_dir).expanduser().resolve()))

    restrict_visible_device_to_local_rank()

    from vidaforge_adapters.vjepa2.patch import apply_vjepa_runtime_patches

    apply_vjepa_runtime_patches()

    params = load_config(Path(args.fname))
    params = with_validation_manifest(params, Path(args.valid_csv))

    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from app.vjepa_2_1.models.utils.masks_dist import compute_mask_distance
    from app.vjepa_2_1.models.utils.modules import Lambda_LinearWarmupHold
    from app.vjepa_2_1.transforms import make_transforms
    from app.vjepa_2_1.utils import init_video_model, normalize_nested
    from src.datasets.data_manager import init_data
    from src.masks.multiseq_multiblock3d import MaskCollator
    from src.masks.utils import apply_masks
    from src.utils.checkpoint_loader import robust_checkpoint_loader
    from src.utils.distributed import init_distributed
    from torch.nn.parallel import DistributedDataParallel

    cfgs_meta = require_mapping(params, "meta")
    cfgs_model = require_mapping(params, "model")
    cfgs_data = require_mapping(params, "data")
    cfgs_data_aug = require_mapping(params, "data_aug")
    cfgs_loss = require_mapping(params, "loss")
    cfgs_mask = params.get("mask")
    if not isinstance(cfgs_mask, list):
        raise ValueError("config key 'mask' must be a list")
    if params.get("img_data") is not None:
        raise NotImplementedError("V-JEPA2 eval adapter currently supports video-only configs")

    seed = int(args.seed if args.seed is not None else cfgs_meta.get("seed", 0))
    rank_env = env_int("RANK", 0) or 0
    set_eval_seed(seed + rank_env)

    world_size, rank = init_distributed()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        pprint.PrettyPrinter(indent=4).pprint(params)

    dtype_name = str(cfgs_meta.get("dtype", "float32")).lower()
    if dtype_name == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif dtype_name == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    model_name = required_value(cfgs_model, "model_name")
    dataset_fpcs = [int(value) for value in required_value(cfgs_data, "dataset_fpcs")]
    max_num_frames = max(dataset_fpcs)
    crop_size = int(cfgs_data.get("crop_size", 224))
    patch_size = int(required_value(cfgs_data, "patch_size"))
    tubelet_size = int(required_value(cfgs_data, "tubelet_size"))
    grid_size = crop_size // patch_size
    batch_size = int(required_value(cfgs_data, "batch_size"))
    fps = required_value(cfgs_data, "fps")
    num_workers = int(cfgs_data.get("num_workers", 1))
    pin_mem = bool(cfgs_data.get("pin_mem", False))
    persistent_workers = bool(cfgs_data.get("persistent_workers", False))

    embed_dim_encoder = encoder_embed_dim(model_name)
    predict_all = bool(cfgs_loss.get("predict_all", True))
    weight_distance_loss = bool(cfgs_loss.get("weight_distance_loss", False))
    offset_context_loss = bool(cfgs_loss.get("offset_context_loss", False))
    loss_exp = float(required_value(cfgs_loss, "loss_exp"))
    shift_by_n = int(cfgs_loss.get("shift_by_n", 0))
    has_cls_first = bool(cfgs_model.get("has_cls_first", False))
    normalize_predictor = bool(cfgs_model.get("normalize_predictor", False))
    levels_predictor = int(cfgs_model.get("levels_predictor", 4))
    lambda_value = float(cfgs_model.get("lambda_value_vid", 0.0))
    lambda_progressive = bool(cfgs_model.get("lambda_progressive", True))

    encoder, predictor = init_video_model(
        uniform_power=bool(cfgs_model.get("uniform_power", False)),
        use_mask_tokens=bool(cfgs_model.get("use_mask_tokens", False)),
        num_mask_tokens=int(len(cfgs_mask) * len(dataset_fpcs)),
        zero_init_mask_tokens=bool(cfgs_model.get("zero_init_mask_tokens", True)),
        device=device,
        patch_size=patch_size,
        max_num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=required_value(cfgs_model, "pred_depth"),
        pred_num_heads=cfgs_model.get("pred_num_heads", None),
        pred_embed_dim=required_value(cfgs_model, "pred_embed_dim"),
        is_causal=bool(cfgs_model.get("is_causal", False)),
        pred_is_causal=bool(cfgs_model.get("pred_is_causal", False)),
        use_sdpa=bool(cfgs_meta.get("use_sdpa", False)),
        use_silu=bool(cfgs_model.get("use_silu", False)),
        use_pred_silu=bool(cfgs_model.get("use_pred_silu", False)),
        wide_silu=bool(cfgs_model.get("wide_silu", True)),
        use_rope=bool(cfgs_model.get("use_rope", False)),
        use_activation_checkpointing=bool(cfgs_model.get("use_activation_checkpointing", False)),
        return_all_tokens=predict_all,
        chop_last_n_tokens=shift_by_n,
        init_type=cfgs_model.get("init_type", "default"),
        img_temporal_dim_size=cfgs_model.get("img_temporal_dim_size", None),
        n_registers=int(cfgs_model.get("n_registers", 0)),
        n_registers_predictor=int(cfgs_model.get("n_registers_predictor", 0)),
        has_cls_first=has_cls_first,
        interpolate_rope=bool(cfgs_model.get("interpolate_rope", False)),
        modality_embedding=bool(cfgs_model.get("modality_embedding", False)),
    )
    target_encoder = copy.deepcopy(encoder)

    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(
        predictor,
        static_graph=False,
        find_unused_parameters=True,
    )
    target_encoder = DistributedDataParallel(target_encoder)
    for param in target_encoder.parameters():
        param.requires_grad = False

    checkpoint = robust_checkpoint_loader(
        str(Path(args.ckpt).expanduser()),
        map_location=torch.device("cpu"),
    )
    checkpoint_epoch = int(checkpoint.get("epoch", 0))
    load_model_state(encoder, checkpoint["encoder"], "encoder")
    load_model_state(predictor, checkpoint["predictor"], "predictor")
    load_model_state(target_encoder, checkpoint["target_encoder"], "target_encoder")
    del checkpoint

    encoder.eval()
    predictor.eval()
    target_encoder.eval()

    mask_collator = MaskCollator(
        cfgs_mask=cfgs_mask,
        dataset_fpcs=dataset_fpcs,
        crop_size=crop_size,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
    )
    transform = make_transforms(
        random_horizontal_flip=True,
        random_resize_aspect_ratio=cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3]),
        random_resize_scale=cfgs_data_aug.get("random_resize_scale", [0.3, 1.0]),
        reprob=float(cfgs_data_aug.get("reprob", 0.0)),
        auto_augment=bool(cfgs_data_aug.get("auto_augment", False)),
        motion_shift=bool(cfgs_data_aug.get("motion_shift", False)),
        crop_size=crop_size,
    )

    loader, sampler = init_data(
        data=cfgs_data.get("dataset_type", "VideoDataset"),
        root_path=required_value(cfgs_data, "datasets"),
        batch_size=batch_size,
        training=False,
        dataset_fpcs=dataset_fpcs,
        fps=fps,
        transform=transform,
        rank=rank,
        world_size=world_size,
        datasets_weights=cfgs_data.get("datasets_weights", None),
        collator=mask_collator,
        num_workers=num_workers,
        pin_mem=pin_mem,
        persistent_workers=persistent_workers,
        drop_last=False,
        log_dir=None,
    )
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(0)

    ipe = int(cfgs_data.get("ipe", params.get("optimization", {}).get("ipe", len(loader))))
    lambda_global_step = checkpoint_epoch * ipe
    if args.lambda_mode == "checkpoint" and lambda_progressive:
        lambda_value_step = float(Lambda_LinearWarmupHold(lambda_value=lambda_value).value(lambda_global_step))
    else:
        lambda_value_step = lambda_value

    local_loss_sum = 0.0
    local_batch_loss_sum = 0.0
    local_sample_count = 0
    local_batch_count = 0

    with torch.no_grad():
        for batch_idx, sample in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            clips, masks_enc, masks_pred, sample_count = load_eval_batch(sample, device)
            if sample_count <= 0:
                continue

            with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision and torch.cuda.is_available()):
                loss = compute_vjepa_loss(
                    encoder=encoder,
                    predictor=predictor,
                    target_encoder=target_encoder,
                    clips=clips,
                    masks_enc=masks_enc,
                    masks_pred=masks_pred,
                    embed_dim_encoder=embed_dim_encoder,
                    levels_predictor=levels_predictor,
                    normalize_predictor=normalize_predictor,
                    predict_all=predict_all,
                    has_cls_first=has_cls_first,
                    loss_exp=loss_exp,
                    grid_size=grid_size,
                    offset_context_loss=offset_context_loss,
                    weight_distance_loss=weight_distance_loss,
                    lambda_value_step=lambda_value_step,
                    apply_masks=apply_masks,
                    compute_mask_distance=compute_mask_distance,
                    F=F,
                    normalize_nested=normalize_nested,
                )

            loss_value = float(loss.detach().float().item())
            local_loss_sum += loss_value * sample_count
            local_batch_loss_sum += loss_value
            local_sample_count += sample_count
            local_batch_count += 1

    stats = torch.tensor(
        [
            local_loss_sum,
            float(local_sample_count),
            local_batch_loss_sum,
            float(local_batch_count),
        ],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    if rank == 0:
        sample_count = int(stats[1].item())
        rank_batch_count = int(stats[3].item())
        result = {
            "checkpoint": str(Path(args.ckpt).expanduser()),
            "checkpoint_epoch": checkpoint_epoch,
            "config_path": str(Path(args.fname).expanduser()),
            "valid_csv": str(Path(args.valid_csv).expanduser()),
            "valid_loss": safe_div(float(stats[0].item()), sample_count),
            "batch_loss_mean": safe_div(float(stats[2].item()), rank_batch_count),
            "sample_count": sample_count,
            "rank_batch_count": rank_batch_count,
            "eval_step_count": safe_div(rank_batch_count, world_size),
            "world_size": world_size,
            "local_batch_size": batch_size,
            "global_batch_size": batch_size * world_size,
            "max_batches": args.max_batches,
            "seed": seed,
            "dtype": dtype_name,
            "model_name": model_name,
            "dataset_fpcs": dataset_fpcs,
            "lambda_mode": args.lambda_mode,
            "lambda_progressive": lambda_progressive,
            "lambda_value": lambda_value,
            "lambda_value_step": lambda_value_step,
            "lambda_global_step": lambda_global_step,
        }
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def load_config(path: Path) -> dict[str, Any]:
    with path.expanduser().open("r", encoding="utf-8") as handle:
        params = yaml.load(handle, Loader=yaml.FullLoader)
    if not isinstance(params, dict):
        raise ValueError(f"V-JEPA2 config must load as a dict: {path}")
    return params


def with_validation_manifest(params: dict[str, Any], valid_csv: Path) -> dict[str, Any]:
    valid_csv = valid_csv.expanduser()
    if not valid_csv.exists():
        raise FileNotFoundError(f"valid CSV does not exist: {valid_csv}")
    output = copy.deepcopy(params)
    data = require_mapping(output, "data")
    data["datasets"] = [str(valid_csv)]
    data["datasets_weights"] = None
    return output


def require_mapping(params: dict[str, Any], key: str) -> dict[str, Any]:
    value = params.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"config key '{key}' must be a mapping")
    return value


def required_value(params: dict[str, Any], key: str) -> Any:
    if key not in params:
        raise ValueError(f"missing required config key '{key}'")
    return params[key]


def set_eval_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encoder_embed_dim(model_name: str) -> int:
    if model_name == "vit_large":
        return 1024
    if model_name == "vit_giant_xformers":
        return 1408
    if model_name == "vit_gigantic_xformers":
        return 1664
    raise ValueError(f"unsupported V-JEPA2 model_name for eval: {model_name}")


def load_model_state(model: Any, checkpoint_state: dict[str, Any], name: str) -> None:
    model_state = model.state_dict()
    checkpoint_state = normalize_state_dict_keys(checkpoint_state, model_state)
    filtered_state = {}
    skipped = []
    for key, value in checkpoint_state.items():
        if key not in model_state:
            skipped.append(key)
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            skipped.append(key)
            continue
        filtered_state[key] = value
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    if skipped:
        print(f"[eval] skipped {len(skipped)} incompatible {name} checkpoint keys")
    if missing:
        print(f"[eval] missing {len(missing)} {name} keys while loading checkpoint")
    if unexpected:
        print(f"[eval] unexpected {len(unexpected)} {name} keys while loading checkpoint")


def normalize_state_dict_keys(
    checkpoint_state: dict[str, Any],
    model_state: dict[str, Any],
) -> dict[str, Any]:
    if not checkpoint_state:
        return checkpoint_state
    checkpoint_keys = set(checkpoint_state)
    model_keys = set(model_state)
    if checkpoint_keys & model_keys:
        return checkpoint_state

    checkpoint_has_module = all(key.startswith("module.") for key in checkpoint_keys)
    model_has_module = all(key.startswith("module.") for key in model_keys)
    if checkpoint_has_module and not model_has_module:
        return {key.removeprefix("module."): value for key, value in checkpoint_state.items()}
    if model_has_module and not checkpoint_has_module:
        return {f"module.{key}": value for key, value in checkpoint_state.items()}
    return checkpoint_state


def load_eval_batch(sample: list[Any], device: Any) -> tuple[list[Any], list[Any], list[Any], int]:
    all_clips, all_masks_enc, all_masks_pred = [], [], []
    sample_count = 0
    for fpc_sample in sample:
        udata, masks_enc, masks_pred = fpc_sample
        clip = udata[0][0].to(device, non_blocking=True)
        sample_count += int(clip.shape[0])
        all_clips.append(clip)
        all_masks_enc.append([mask.to(device, non_blocking=True) for mask in masks_enc])
        all_masks_pred.append([mask.to(device, non_blocking=True) for mask in masks_pred])
    return all_clips, all_masks_enc, all_masks_pred, sample_count


def compute_vjepa_loss(
    *,
    encoder: Any,
    predictor: Any,
    target_encoder: Any,
    clips: list[Any],
    masks_enc: list[Any],
    masks_pred: list[Any],
    embed_dim_encoder: int,
    levels_predictor: int,
    normalize_predictor: bool,
    predict_all: bool,
    has_cls_first: bool,
    loss_exp: float,
    grid_size: int,
    offset_context_loss: bool,
    weight_distance_loss: bool,
    lambda_value_step: float,
    apply_masks: Any,
    compute_mask_distance: Any,
    F: Any,
    normalize_nested: Any,
) -> Any:
    h = forward_target(
        target_encoder=target_encoder,
        clips=clips,
        embed_dim_encoder=embed_dim_encoder,
        levels_predictor=levels_predictor,
        F=F,
    )
    z_pred, z_context = forward_context(
        encoder=encoder,
        predictor=predictor,
        clips=clips,
        masks_enc=masks_enc,
        masks_pred=masks_pred,
        embed_dim_encoder=embed_dim_encoder,
        normalize_predictor=normalize_predictor,
        predict_all=predict_all,
        normalize_nested=normalize_nested,
    )
    loss = loss_fn(
        z=z_pred,
        h=h,
        masks_to_apply=masks_pred,
        cls_loss=has_cls_first,
        d_weights=None,
        loss_exp=loss_exp,
        apply_masks=apply_masks,
    )
    if predict_all:
        distance_weights = compute_mask_distance(
            masks_pred,
            masks_enc,
            grid_size,
            offset_context_loss,
        )
        d_weights = distance_weights if weight_distance_loss else None
        loss_context = loss_fn(
            z=z_context,
            h=h,
            masks_to_apply=masks_enc,
            cls_loss=False,
            d_weights=d_weights,
            loss_exp=loss_exp,
            apply_masks=apply_masks,
        )
        loss = loss + loss_context * lambda_value_step
    return loss


def forward_target(
    *,
    target_encoder: Any,
    clips: list[Any],
    embed_dim_encoder: int,
    levels_predictor: int,
    F: Any,
) -> list[Any]:
    h = target_encoder(clips, gram_mode=False, training_mode=True)
    output = []
    for hi in h:
        if levels_predictor > 1:
            chunks = [
                F.layer_norm(hi[:, :, :embed_dim_encoder], (embed_dim_encoder,)),
                F.layer_norm(
                    hi[:, :, embed_dim_encoder : embed_dim_encoder * 2],
                    (embed_dim_encoder,),
                ),
                F.layer_norm(
                    hi[:, :, embed_dim_encoder * 2 : embed_dim_encoder * 3],
                    (embed_dim_encoder,),
                ),
                F.layer_norm(hi[:, :, -embed_dim_encoder:], (embed_dim_encoder,)),
            ]
            output.append(torch_cat(chunks, dim=2))
        else:
            output.append(F.layer_norm(hi, (hi.size(-1),)))
    return output


def forward_context(
    *,
    encoder: Any,
    predictor: Any,
    clips: list[Any],
    masks_enc: list[Any],
    masks_pred: list[Any],
    embed_dim_encoder: int,
    normalize_predictor: bool,
    predict_all: bool,
    normalize_nested: Any,
) -> tuple[Any, Any]:
    z = encoder(clips, masks_enc, gram_mode=False, training_mode=True)
    z_pred, z_context = predictor(z, masks_enc, masks_pred, mod="video")
    if normalize_predictor:
        z_pred = normalize_nested(z_pred, embed_dim_encoder)
        if predict_all:
            z_context = normalize_nested(z_context, embed_dim_encoder)
    return z_pred, z_context


def loss_fn(
    *,
    z: list[Any],
    h: list[Any],
    masks_to_apply: list[Any],
    cls_loss: bool,
    d_weights: list[Any] | None,
    loss_exp: float,
    apply_masks: Any,
) -> Any:
    if cls_loss:
        h_cls = [hi[:, 0].unsqueeze(1) for hi in h]
        masked_h = [
            apply_masks(hi[:, 1:], mi, concat=False)
            for hi, mi in zip(h, masks_to_apply, strict=True)
        ]
        loss, count = 0, 0
        for zi, hi, hi_cls in zip(z, masked_h, h_cls, strict=True):
            for zij, hij in zip(zi, hi, strict=True):
                h_term = torch_cat([hi_cls, hij], dim=1)
                loss += mean_abs_power(zij, h_term, loss_exp)
                count += 1
        return loss / count

    masked_h = [
        apply_masks(hi, mi, concat=False)
        for hi, mi in zip(h, masks_to_apply, strict=True)
    ]
    loss, count = 0, 0
    if d_weights is not None:
        for zi, hi, d_i in zip(z, masked_h, d_weights, strict=True):
            for zij, hij, d_ij in zip(zi, hi, d_i, strict=True):
                loss_n = abs_power(zij, hij, loss_exp) * (1 / d_ij.unsqueeze(2))
                loss += loss_n.mean() / loss_exp
                count += 1
    else:
        for zi, hi in zip(z, masked_h, strict=True):
            for zij, hij in zip(zi, hi, strict=True):
                loss += mean_abs_power(zij, hij, loss_exp)
                count += 1
    return loss / count


def abs_power(left: Any, right: Any, loss_exp: float) -> Any:
    return abs(left - right) ** loss_exp


def mean_abs_power(left: Any, right: Any, loss_exp: float) -> Any:
    return abs_power(left, right, loss_exp).mean() / loss_exp


def torch_cat(values: list[Any], dim: int) -> Any:
    import torch

    return torch.cat(values, dim=dim)


def safe_div(numerator: float | int, denominator: float | int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def restrict_visible_device_to_local_rank() -> None:
    if os.environ.get("VJEPA_SET_CUDA_VISIBLE_DEVICES", "1") == "0":
        return

    local_rank = env_int("LOCAL_RANK", env_int("SLURM_LOCALID", 0))
    if local_rank is None:
        local_rank = 0

    current = os.environ.get("CUDA_VISIBLE_DEVICES")
    if current:
        visible_devices = [item.strip() for item in current.split(",") if item.strip()]
        if len(visible_devices) == 1:
            return
        if 0 <= local_rank < len(visible_devices):
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices[local_rank]
            return

    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)


def env_int(name: str, fallback: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return fallback
    return int(value)


if __name__ == "__main__":
    main()
