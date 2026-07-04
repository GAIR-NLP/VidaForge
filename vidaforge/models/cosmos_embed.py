from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _find_pruneable_heads_and_indices(
    heads: list[int],
    n_heads: int,
    head_size: int,
    already_pruned_heads: set[int],
) -> tuple[set[int], torch.LongTensor]:
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for head in heads:
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()
    return heads, index


def _get_head_mask(
    self: Any,
    head_mask: torch.Tensor | None,
    num_hidden_layers: int,
    is_attention_chunked: bool = False,
) -> torch.Tensor | list[None]:
    if head_mask is None:
        return [None] * num_hidden_layers

    if head_mask.dim() == 1:
        head_mask = head_mask[None, None, :, None, None]
        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.dim() == 2:
        head_mask = head_mask[:, None, :, None, None]
    if head_mask.dim() != 5:
        raise ValueError(f"head_mask.dim must be 5, got {head_mask.dim()}.")

    head_mask = head_mask.to(dtype=self.dtype)
    if is_attention_chunked:
        head_mask = head_mask.unsqueeze(-1)
    return head_mask


def _ensure_transformers5_loading_attrs(model: Any) -> None:
    if not hasattr(model, "all_tied_weights_keys"):
        model.all_tied_weights_keys = model.get_expanded_tied_weights_keys(
            all_submodels=False
        )
    for attr_name in ("_tp_plan", "_ep_plan", "_pp_plan"):
        if getattr(model, attr_name, None) is None:
            setattr(model, attr_name, {})
    for attr_name in (
        "_keep_in_fp32_modules",
        "_keep_in_fp32_modules_strict",
        "_no_split_modules",
        "_skip_keys_device_placement",
        "_keys_to_ignore_on_load_unexpected",
        "_keys_to_ignore_on_load_missing",
        "_keys_to_ignore_on_save",
    ):
        if getattr(model, attr_name, None) is None:
            setattr(model, attr_name, set())


def patch_transformers_for_cosmos_embed() -> None:
    """Patch current-process transformers APIs expected by Cosmos-Embed1 code."""
    from transformers import PreTrainedModel
    import transformers.pytorch_utils as transformers_pytorch_utils

    if not hasattr(transformers_pytorch_utils, "find_pruneable_heads_and_indices"):
        transformers_pytorch_utils.find_pruneable_heads_and_indices = (
            _find_pruneable_heads_and_indices
        )

    if not hasattr(PreTrainedModel, "get_head_mask"):
        PreTrainedModel.get_head_mask = _get_head_mask

    if getattr(PreTrainedModel, "_cosmos_embed_compat_patched", False):
        return

    original_tie_weights = PreTrainedModel.tie_weights
    original_move_missing_keys = PreTrainedModel._move_missing_keys_from_meta_to_device

    def patched_tie_weights(self: Any, *args: Any, **kwargs: Any) -> Any:
        _ensure_transformers5_loading_attrs(self)
        return original_tie_weights(self, *args, **kwargs)

    def patched_move_missing_keys(self: Any, *args: Any, **kwargs: Any) -> Any:
        _ensure_transformers5_loading_attrs(self)
        return original_move_missing_keys(self, *args, **kwargs)

    PreTrainedModel.tie_weights = patched_tie_weights
    PreTrainedModel._move_missing_keys_from_meta_to_device = patched_move_missing_keys
    PreTrainedModel._cosmos_embed_compat_patched = True


def _resolve_model_path(
    model_name_or_path: str | Path,
    *,
    local_files_only: bool,
) -> Path:
    model_path = Path(model_name_or_path)
    if model_path.exists():
        return model_path

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=str(model_name_or_path),
            local_files_only=local_files_only,
        )
    )


def _load_safetensors_state_dict(model_path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    shard_paths = sorted(model_path.glob("model-*.safetensors"))
    if not shard_paths:
        shard_paths = sorted(model_path.glob("*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(f"No safetensors checkpoint found in {model_path}.")

    state_dict: dict[str, torch.Tensor] = {}
    for shard_path in shard_paths:
        state_dict.update(load_file(shard_path, device="cpu"))
    return state_dict


def load_cosmos_embed_model(
    model_name_or_path: str | Path,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    local_files_only: bool = False,
) -> Any:
    """Load Cosmos-Embed1 with compatibility patches for transformers 5.x.

    Cosmos-Embed1 remote code targets transformers 4.51.3. This loader avoids
    the transformers 5.x safetensors path that skips the model's positional
    embedding interpolation hook.
    """
    from transformers import AutoConfig, AutoModel

    patch_transformers_for_cosmos_embed()
    model_path = _resolve_model_path(
        model_name_or_path,
        local_files_only=local_files_only,
    )
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model = AutoModel.from_config(config, trust_remote_code=True)
    _ensure_transformers5_loading_attrs(model)

    state_dict = _load_safetensors_state_dict(model_path)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Cosmos-Embed1 compat load failed with "
            f"{len(missing_keys)} missing keys and "
            f"{len(unexpected_keys)} unexpected keys."
        )
    if device is not None or dtype is not None:
        model = model.to(device=device, dtype=dtype)
    return model


def load_cosmos_embed_processor(
    model_name_or_path: str | Path,
    *,
    local_files_only: bool = False,
) -> Any:
    from transformers import AutoProcessor

    patch_transformers_for_cosmos_embed()
    model_path = _resolve_model_path(
        model_name_or_path,
        local_files_only=local_files_only,
    )
    return AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
