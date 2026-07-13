from __future__ import annotations

import argparse
import math
import os
import pprint
import sys
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "torchrun/k8s launcher for V-JEPA2 pretraining with VidaForge "
            "TorchCodec dataset patches."
        )
    )
    parser.add_argument(
        "--fname",
        required=True,
        help="V-JEPA2 YAML config file to load.",
    )
    parser.add_argument(
        "--vjepa2-dir",
        dest="vjepa2_dir",
        default=None,
        help="Optional path to the V-JEPA2 repository. Prepended to sys.path before imports.",
    )
    parser.add_argument(
        "--resume-preempt",
        action="store_true",
        help="Pass resume_preempt=True to the official V-JEPA2 app scaffold.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help=(
            "OmegaConf dot-list overrides applied after loading the YAML, for example "
            "folder=/path/to/output data.datasets=[/path/to/train.csv]."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.vjepa2_dir:
        sys.path.insert(0, str(Path(args.vjepa2_dir).expanduser().resolve()))

    _restrict_visible_device_to_local_rank()

    from vidaforge_adapters.vjepa2.patch import apply_vjepa_runtime_patches

    apply_vjepa_runtime_patches()

    params = _load_config(Path(args.fname), args.overrides)
    _apply_runtime_topology(params)
    _resolve_auto_ipe(params)
    rank = _env_int("RANK", 0)
    folder = Path(params["folder"]).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        pprint.PrettyPrinter(indent=4).pprint(params)
        params_path = folder / "params-pretrain.yaml"
        params_path.write_text(
            yaml.dump(params),
            encoding="utf-8",
        )

    from app.scaffold import main as app_main

    app_main(params["app"], args=params, resume_preempt=args.resume_preempt)


def _load_config(path: Path, overrides: list[str] | None = None) -> dict[str, Any]:
    config = OmegaConf.load(path.expanduser())
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    params = OmegaConf.to_container(config, resolve=True)
    if not isinstance(params, dict):
        raise ValueError(f"V-JEPA2 config must load as a dict: {path}")
    if "app" not in params:
        raise ValueError(f"V-JEPA2 config missing required key 'app': {path}")
    if "folder" not in params:
        raise ValueError(f"V-JEPA2 config missing required key 'folder': {path}")
    return params


def _apply_runtime_topology(params: dict[str, Any]) -> None:
    world_size = _env_int("WORLD_SIZE")
    local_world_size = _env_int("LOCAL_WORLD_SIZE")
    if world_size is None:
        return

    if local_world_size is None or local_world_size <= 0:
        local_world_size = world_size
    params["nodes"] = max(1, math.ceil(world_size / local_world_size))
    params["tasks_per_node"] = local_world_size


def _resolve_auto_ipe(params: dict[str, Any]) -> None:
    optimization = _required_mapping(params, "optimization")
    if optimization.get("ipe") != "auto":
        return

    data = _required_mapping(params, "data")
    datasets = data.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("data.datasets must be a non-empty list when optimization.ipe=auto")

    sample_count = sum(_count_csv_manifest_rows(path) for path in datasets)
    if sample_count <= 0:
        raise ValueError("data.datasets contain no training samples")

    batch_size = int(data.get("batch_size", 0))
    dataset_epochs = int(optimization.get("epochs", 0))
    world_size = _env_int("WORLD_SIZE", 1) or 1
    if batch_size <= 0:
        raise ValueError("data.batch_size must be > 0 when optimization.ipe=auto")
    if dataset_epochs <= 0:
        raise ValueError("optimization.epochs must be > 0 when optimization.ipe=auto")

    steps_per_dataset_epoch = math.ceil(sample_count / (world_size * batch_size))
    optimization["ipe"] = max(1, steps_per_dataset_epoch)
    params["vidaforge_runtime"] = {
        "dataset_epochs": dataset_epochs,
        "sample_count": sample_count,
        "world_size": world_size,
        "per_process_batch_size": batch_size,
        "global_batch_size": world_size * batch_size,
        "total_steps": dataset_epochs * optimization["ipe"],
        "resolved_ipe": optimization["ipe"],
    }


def _required_mapping(params: dict[str, Any], key: str) -> dict[str, Any]:
    value = params.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"V-JEPA2 config key '{key}' must be a mapping")
    return value


def _count_csv_manifest_rows(path: str | Path) -> int:
    manifest_path = Path(path).expanduser()
    if manifest_path.suffix.lower() != ".csv":
        raise ValueError(
            "optimization.ipe=auto currently requires data.datasets to contain "
            f"CSV manifests, got: {manifest_path}"
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"V-JEPA2 manifest does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _restrict_visible_device_to_local_rank() -> None:
    if os.environ.get("VJEPA_SET_CUDA_VISIBLE_DEVICES", "1") == "0":
        return

    local_rank = _env_int("LOCAL_RANK", _env_int("SLURM_LOCALID", 0))
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


def _env_int(name: str, fallback: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return fallback
    return int(value)


if __name__ == "__main__":
    main()
