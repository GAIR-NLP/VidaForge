from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from .sampling import compute_vjepa_indices


@dataclass(frozen=True)
class VJEPASample:
    path: str
    label: Any
    dataset_index: int


def _coerce_label(value: str) -> Any:
    value = value.strip()
    if not value:
        return 0
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _path_from_npy_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    text = repr(value)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return str(value)


def read_vjepa_manifest(path: str | Path) -> list[tuple[str, Any]]:
    """Read a V-JEPA `.csv` or `.npy` manifest.

    CSV parsing follows the official dataset convention: the first field is the
    path and the second field is the label. Plain whitespace and `::` are both
    accepted as separators.
    """
    manifest_path = Path(path).expanduser()
    suffix = manifest_path.suffix.lower()
    if suffix == ".csv":
        items: list[tuple[str, Any]] = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                if "::" in line:
                    parts = line.split("::", 1)
                else:
                    parts = line.split(maxsplit=1)
                if not parts or not parts[0].strip():
                    raise ValueError(f"missing video path at {manifest_path}:{line_no}")
                label = _coerce_label(parts[1]) if len(parts) > 1 else 0
                items.append((parts[0].strip(), label))
        return items

    if suffix == ".npy":
        data = np.load(manifest_path, allow_pickle=True)
        return [(_path_from_npy_value(value), 0) for value in data]

    return [(str(manifest_path), 0)]


class TorchCodecVideoDataset(torch.utils.data.Dataset):
    """V-JEPA VideoDataset-compatible loader backed by TorchCodec.

    The returned item matches the official dataset contract:
    `(buffer, label, clip_indices)`, where `buffer` is a list of clips after
    optional transforms and `clip_indices` is a list of numpy integer arrays.
    """

    def __init__(
        self,
        data_paths: str | Sequence[str],
        datasets_weights: Sequence[float] | None = None,
        frames_per_clip: int | None = 16,
        fps: int | None = None,
        dataset_fpcs: Sequence[int] | None = None,
        frame_step: int | None = 4,
        num_clips: int = 1,
        transform: Callable[[Any], Any] | None = None,
        shared_transform: Callable[[Any], Any] | None = None,
        random_clip_sampling: bool = True,
        allow_clip_overlap: bool = False,
        filter_short_videos: bool = False,
        filter_long_videos: int = int(10**9),
        duration: float | None = None,
        torchcodec_seek_mode: str = "exact",
        torchcodec_device: str = "cpu",
        num_ffmpeg_threads: int = 1,
        max_decode_retries: int = 10,
    ) -> None:
        specified = sum(value is not None for value in (fps, duration, frame_step))
        if specified != 1:
            raise ValueError(
                f"Must specify exactly one of either {fps=}, {duration=}, or {frame_step=}."
            )
        if num_clips <= 0:
            raise ValueError("num_clips must be > 0")
        if max_decode_retries <= 0:
            raise ValueError("max_decode_retries must be > 0")

        if isinstance(data_paths, str):
            data_path_list = [data_paths]
        else:
            data_path_list = [str(path) for path in data_paths]
        if not data_path_list:
            raise ValueError("data_paths must not be empty")

        if dataset_fpcs is None:
            if frames_per_clip is None:
                raise ValueError("frames_per_clip must be set when dataset_fpcs is not set")
            if frames_per_clip <= 0:
                raise ValueError("frames_per_clip must be > 0")
            self.dataset_fpcs = [frames_per_clip for _ in data_path_list]
        else:
            if len(dataset_fpcs) != len(data_path_list):
                raise ValueError("dataset_fpcs length must match data_paths length")
            self.dataset_fpcs = [int(value) for value in dataset_fpcs]
            if any(value <= 0 for value in self.dataset_fpcs):
                raise ValueError("dataset_fpcs values must be > 0")
            if frames_per_clip is not None and frames_per_clip <= 0:
                raise ValueError("frames_per_clip must be > 0")

        self.data_paths = data_path_list
        self.datasets_weights = list(datasets_weights) if datasets_weights is not None else None
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.transform = transform
        self.shared_transform = shared_transform
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.filter_short_videos = filter_short_videos
        self.filter_long_videos = filter_long_videos
        self.duration = duration
        self.fps = fps
        self.torchcodec_seek_mode = torchcodec_seek_mode
        self.torchcodec_device = torchcodec_device
        self.num_ffmpeg_threads = num_ffmpeg_threads
        self.max_decode_retries = max_decode_retries

        samples: list[VJEPASample] = []
        self.num_samples_per_dataset: list[int] = []
        for dataset_index, data_path in enumerate(self.data_paths):
            manifest_items = read_vjepa_manifest(data_path)
            self.num_samples_per_dataset.append(len(manifest_items))
            for sample_path, label in manifest_items:
                samples.append(
                    VJEPASample(
                        path=sample_path,
                        label=label,
                        dataset_index=dataset_index,
                    )
                )
        if not samples:
            raise ValueError(f"no samples found in data_paths={self.data_paths}")

        self.records = samples
        self.samples = [sample.path for sample in samples]
        self.labels = [sample.label for sample in samples]

        self.sample_weights: list[float] | None = None
        if self.datasets_weights is not None:
            if len(self.datasets_weights) != len(self.num_samples_per_dataset):
                raise ValueError("datasets_weights length must match data_paths length")
            self.sample_weights = []
            for dataset_weight, sample_count in zip(
                self.datasets_weights,
                self.num_samples_per_dataset,
                strict=True,
            ):
                if sample_count <= 0:
                    raise ValueError("cannot assign a dataset weight to an empty dataset")
                self.sample_weights.extend([float(dataset_weight) / sample_count] * sample_count)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[list[Any], Any, list[np.ndarray]]:
        last_error: str | None = None
        for _ in range(self.max_decode_retries):
            record = self.records[index]
            if not isinstance(record.path, str):
                last_error = f"invalid sample path type: {type(record.path)}"
                warnings.warn(last_error)
                loaded_sample = None
            elif _is_image_path(record.path):
                loaded_sample = self.get_item_image(index)
            else:
                loaded_sample = self.get_item_video(index)

            if loaded_sample is not None:
                return loaded_sample

            index = int(np.random.randint(len(self)))

        raise RuntimeError(
            "failed to decode a valid V-JEPA sample after "
            f"{self.max_decode_retries} retries: {last_error}"
        )

    def get_item_video(self, index: int) -> tuple[list[Any], Any, list[np.ndarray]] | None:
        record = self.records[index]
        frames_per_clip = self.dataset_fpcs[record.dataset_index]

        buffer, clip_indices = self.loadvideo_torchcodec(record.path, frames_per_clip)
        if len(buffer) <= 0:
            return None

        def split_into_clips(video: Any) -> list[Any]:
            return [
                video[clip_idx * frames_per_clip : (clip_idx + 1) * frames_per_clip]
                for clip_idx in range(self.num_clips)
            ]

        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)
        clips = split_into_clips(buffer)
        if self.transform is not None:
            clips = [self.transform(clip) for clip in clips]

        return clips, record.label, clip_indices

    def get_item_image(self, index: int) -> tuple[list[Any], Any, list[np.ndarray]] | None:
        try:
            import torchvision
        except ImportError as exc:  # pragma: no cover - depends on training env
            raise ImportError("torchvision is required for V-JEPA image samples") from exc

        record = self.records[index]
        frames_per_clip = self.dataset_fpcs[record.dataset_index]
        sample_path = Path(record.path).expanduser()
        try:
            image_tensor = torchvision.io.read_image(
                path=str(sample_path),
                mode=torchvision.io.ImageReadMode.RGB,
            )
        except Exception as exc:
            warnings.warn(f"failed to read image sample {sample_path}: {type(exc).__name__}: {exc}")
            return None

        clip_indices = [np.arange(start=0, stop=frames_per_clip, dtype=np.int32)]
        buffer = image_tensor.unsqueeze(dim=0).repeat((frames_per_clip, 1, 1, 1))
        buffer = buffer.permute((0, 2, 3, 1))

        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)
        clips: list[Any] = [buffer]
        if self.transform is not None:
            clips = [self.transform(buffer)]

        return clips, record.label, clip_indices

    def loadvideo_torchcodec(self, sample: str, frames_per_clip: int) -> tuple[np.ndarray, list[np.ndarray]]:
        sample_path = Path(sample).expanduser()
        if not sample_path.exists():
            warnings.warn(f"video path not found sample={sample}")
            return np.empty((0,), dtype=np.uint8), []

        file_size = sample_path.stat().st_size
        if file_size > self.filter_long_videos:
            warnings.warn(f"skipping long video of size file_size={file_size} bytes")
            return np.empty((0,), dtype=np.uint8), []

        try:
            from torchcodec.decoders import VideoDecoder, set_cuda_backend

            cuda_backend = (
                set_cuda_backend("beta")
                if self.torchcodec_device.startswith("cuda")
                else nullcontext()
            )
            with cuda_backend:
                decoder = VideoDecoder(
                    sample_path,
                    device=self.torchcodec_device,
                    dimension_order="NCHW",
                    seek_mode=self.torchcodec_seek_mode,
                    num_ffmpeg_threads=self.num_ffmpeg_threads,
                )
                total_frames = int(len(decoder))
                metadata = decoder.metadata
                video_fps = float(metadata.average_fps or 0.0)
                all_indices, clip_indices, _ = compute_vjepa_indices(
                    total_frames=total_frames,
                    video_fps=video_fps,
                    frames_per_clip=frames_per_clip,
                    frame_step=self.frame_step,
                    duration=self.duration,
                    fps=self.fps,
                    num_clips=self.num_clips,
                    random_clip_sampling=self.random_clip_sampling,
                    allow_clip_overlap=self.allow_clip_overlap,
                    filter_short_videos=self.filter_short_videos,
                )
                frames = decoder.get_frames_at(all_indices).data.contiguous()
        except Exception as exc:
            warnings.warn(
                f"failed to read video sample {sample_path}: "
                f"{type(exc).__name__}: {exc}"
            )
            return np.empty((0,), dtype=np.uint8), []

        return frames.cpu().permute(0, 2, 3, 1).numpy(), clip_indices


class DistributedWeightedSampler(DistributedSampler):
    """Small local copy of V-JEPA's weighted distributed sampler behavior."""

    def __init__(
        self,
        dataset: TorchCodecVideoDataset,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if dataset.sample_weights is None:
            raise ValueError(
                "dataset.sample_weights must be set for DistributedWeightedSampler"
            )
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

    @property
    def sample_probabilities(self) -> np.ndarray:
        dataset = self.dataset
        if not isinstance(dataset, TorchCodecVideoDataset) or dataset.sample_weights is None:
            raise TypeError(
                "DistributedWeightedSampler requires TorchCodecVideoDataset.sample_weights"
            )
        sample_weights = np.asarray(dataset.sample_weights, dtype=np.float64)
        weight_sum = float(np.sum(sample_weights))
        if weight_sum <= 0:
            raise ValueError("sample weights must sum to a positive value")
        return sample_weights / weight_sum

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        indices = rng.choice(
            range(0, len(self.dataset)),
            size=self.total_size,
            p=self.sample_probabilities,
            replace=True,
        ).tolist()

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            indices = indices[: self.total_size]

        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(int(index) for index in indices)


def make_videodataset(
    data_paths: str | Sequence[str],
    batch_size: int,
    frames_per_clip: int | None = 8,
    dataset_fpcs: Sequence[int] | None = None,
    frame_step: int | None = 4,
    duration: float | None = None,
    fps: int | None = None,
    num_clips: int = 1,
    random_clip_sampling: bool = True,
    allow_clip_overlap: bool = False,
    filter_short_videos: bool = False,
    filter_long_videos: int = int(10**9),
    transform: Callable[[Any], Any] | None = None,
    shared_transform: Callable[[Any], Any] | None = None,
    rank: int = 0,
    world_size: int = 1,
    datasets_weights: Sequence[float] | None = None,
    collator: Callable[[Any], Any] | None = None,
    drop_last: bool = True,
    num_workers: int = 10,
    pin_mem: bool = True,
    persistent_workers: bool = True,
    deterministic: bool = True,
    log_dir: str | Path | None = None,
    torchcodec_seek_mode: str = "exact",
    torchcodec_device: str = "cpu",
    num_ffmpeg_threads: int = 1,
    max_decode_retries: int = 10,
) -> tuple[TorchCodecVideoDataset, DataLoader, DistributedSampler]:
    """Build a V-JEPA-compatible dataset/dataloader pair using TorchCodec."""
    if log_dir is not None:
        warnings.warn(
            "log_dir resource monitoring is not implemented in the TorchCodec V-JEPA adapter"
        )
    if not deterministic:
        warnings.warn(
            "deterministic=False is ignored; the TorchCodec V-JEPA adapter uses torch DataLoader"
        )

    dataset = TorchCodecVideoDataset(
        data_paths=data_paths,
        datasets_weights=datasets_weights,
        frames_per_clip=frames_per_clip,
        dataset_fpcs=dataset_fpcs,
        duration=duration,
        fps=fps,
        frame_step=frame_step,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_short_videos=filter_short_videos,
        filter_long_videos=filter_long_videos,
        shared_transform=shared_transform,
        transform=transform,
        torchcodec_seek_mode=torchcodec_seek_mode,
        torchcodec_device=torchcodec_device,
        num_ffmpeg_threads=num_ffmpeg_threads,
        max_decode_retries=max_decode_retries,
    )

    if datasets_weights is not None:
        dist_sampler: DistributedSampler = DistributedWeightedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
    else:
        dist_sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )

    data_loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "collate_fn": collator,
        "sampler": dist_sampler,
        "batch_size": batch_size,
        "drop_last": drop_last,
        "pin_memory": pin_mem,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        data_loader_kwargs["persistent_workers"] = persistent_workers

    data_loader = DataLoader(**data_loader_kwargs)
    return dataset, data_loader, dist_sampler


def _is_image_path(path: str) -> bool:
    return Path(path).suffix.lower() in {".jpg", ".jpeg", ".png"}
