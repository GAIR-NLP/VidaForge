from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from vidaforge.common import join_data_dir, parse_json_object
from vidaforge.media.frames import (
    iter_rgb_frame_tensor_batches,
    load_rgb_frame_tensors,
    select_uniform_items,
)
from vidaforge.models.cosmos_embed import (
    load_cosmos_embed_model,
    load_cosmos_embed_processor,
)

from .base import DeduplicatorBase
from .config import CosmosDedupConfig
from .faiss_index import FaissIndex, build_flat_ip_index
from .grouping import merge_duplicate_pairs

_COSMOS_FRAME_COUNT = 8


@dataclass(frozen=True, slots=True)
class CosmosFeature:
    clip_id: str
    ok: int
    error: str
    input_frame_count: int
    selected_frame_count: int
    embedding: np.ndarray

    @property
    def has_embedding(self) -> bool:
        return int(self.ok) == 1 and self.embedding.size > 0


@dataclass(frozen=True, slots=True)
class CosmosClipInput:
    row_index: int
    clip_id: str
    frame_count: int
    frame_paths: list[Path]


class CosmosVideoEmbeddingModel:
    def __init__(self, model_name: str, *, frame_load_workers: int = 1) -> None:
        if frame_load_workers <= 0:
            raise ValueError("frame_load_workers must be > 0")
        self.model_name = model_name
        self.frame_load_workers = int(frame_load_workers)
        self.processor: Any | None = None
        self.model: Any | None = None
        self.device: Any | None = None
        self.dtype: Any | None = None

    def embed(self, frame_batches: list[list[Path]]) -> np.ndarray:
        if not frame_batches:
            return np.empty((0, 0), dtype=np.float32)
        rgb_videos = self._load_rgb_video_tensors(frame_batches)
        return self.embed_rgb_videos(rgb_videos)

    def embed_rgb_videos(self, rgb_videos: list[Any]) -> np.ndarray:
        if not rgb_videos:
            return np.empty((0, 0), dtype=np.float32)
        self._load()
        shapes = {video.shape for video in rgb_videos}
        if len(shapes) == 1:
            return self._embed_rgb_videos(rgb_videos)
        return np.ascontiguousarray(
            np.vstack([self._embed_rgb_videos([video]) for video in rgb_videos]),
            dtype=np.float32,
        )

    @staticmethod
    def stack_rgb_video_tensors(frame_batch: list[Any]) -> list[Any]:
        if len(frame_batch) % _COSMOS_FRAME_COUNT != 0:
            raise RuntimeError(
                "Cosmos frame batch size must be divisible by "
                f"{_COSMOS_FRAME_COUNT}, got {len(frame_batch)}"
            )

        import torch

        return [
            torch.stack(
                frame_batch[start : start + _COSMOS_FRAME_COUNT],
                dim=0,
            ).contiguous()
            for start in range(0, len(frame_batch), _COSMOS_FRAME_COUNT)
        ]

    def _load_rgb_video_tensors(self, frame_batches: list[list[Path]]) -> list[Any]:
        frame_counts = [len(frame_paths) for frame_paths in frame_batches]
        flat_frame_paths = [
            frame_path
            for frame_paths in frame_batches
            for frame_path in frame_paths
        ]
        flat_frames = load_rgb_frame_tensors(
            flat_frame_paths,
            max_workers=self.frame_load_workers,
        )

        import torch

        videos: list[Any] = []
        offset = 0
        for frame_count in frame_counts:
            frames = flat_frames[offset : offset + frame_count]
            offset += frame_count
            videos.append(torch.stack(frames, dim=0).contiguous())
        return videos

    def _load(self) -> None:
        if self.model is not None and self.processor is not None:
            return

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CosmosDeduplicator requires a CUDA GPU for inference.")

        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        self.model = load_cosmos_embed_model(
            self.model_name,
            device=self.device,
            dtype=self.dtype,
        )
        self.model.eval()
        self.processor = load_cosmos_embed_processor(self.model_name)

    def _embed_rgb_videos(self, rgb_videos: list[Any]) -> np.ndarray:
        if self.model is None or self.processor is None:
            raise RuntimeError("model must be loaded before embedding.")

        import torch

        video_batch = torch.stack(rgb_videos, dim=0)
        video_batch = video_batch.to(self.device, non_blocking=True)
        inputs = self.processor(videos=video_batch).to(self.device, dtype=self.dtype)
        with torch.inference_mode():
            output = self.model.get_video_embeddings(**inputs)
        embeddings = (
            output.visual_proj
            if hasattr(output, "visual_proj")
            else output["visual_proj"]
        )
        return normalize_embedding(
            embeddings.detach().float().cpu().numpy()
        )


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    array = np.asarray(embedding, dtype=np.float32)
    if array.ndim == 1:
        norm = np.linalg.norm(array)
        if norm <= 0.0:
            raise ValueError("embedding contains zero vector")
        return np.ascontiguousarray(array / norm, dtype=np.float32)
    if array.ndim == 2:
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        if np.any(norms <= 0.0):
            raise ValueError("embeddings contain zero vectors")
        return np.ascontiguousarray(array / norms, dtype=np.float32)
    raise ValueError(f"expected 1D or 2D embedding, got shape {array.shape}")


def is_better_cosmos_match(
    candidate: dict[str, object],
    current: dict[str, object],
) -> bool:
    return float(candidate["cosine_similarity"]) > float(current["cosine_similarity"])


class CosmosDeduplicator(DeduplicatorBase[CosmosDedupConfig]):
    deduplicator_name: ClassVar[str] = "cosmos"
    config_type: ClassVar[type[CosmosDedupConfig]] = CosmosDedupConfig

    def __init__(
        self,
        config: CosmosDedupConfig,
        *,
        use_gpu_faiss: bool = False,
    ) -> None:
        super().__init__(config, use_gpu_faiss=use_gpu_faiss)
        self.features: list[CosmosFeature] = []
        self.embedding_dim = 0
        self.embeddings: np.ndarray | None = None
        self.embedding_feature_indices: np.ndarray | None = None
        self.faiss_embedding: FaissIndex | None = None
        self.faiss_embedding_index: Any | None = None
        self._embedding_model: CosmosVideoEmbeddingModel | None = None

    def apply(self, row: dict[str, object]) -> None:
        self.apply_batch([row])

    def apply_batch(self, rows: list[dict[str, object]]) -> None:
        clip_inputs: list[CosmosClipInput] = []
        features: list[CosmosFeature | None] = [None] * len(rows)

        for row_index, row in enumerate(rows):
            clip_id = str(row["clip_id"])
            try:
                frame_json = parse_json_object(row["frame_json"], description="frame_json")
                if frame_json is None:
                    raise ValueError("frame_json is missing")
                frame_paths_value = frame_json["frame_paths"]
                if not isinstance(frame_paths_value, list):
                    raise ValueError("frame_json.frame_paths must be a list")
                frame_paths = [join_data_dir(str(path)) for path in frame_paths_value]
                cosmos_frame_paths = select_uniform_items(
                    frame_paths,
                    target_count=_COSMOS_FRAME_COUNT,
                    description="frame_json.frame_paths",
                )
                clip_inputs.append(
                    CosmosClipInput(
                        row_index=row_index,
                        clip_id=clip_id,
                        frame_count=len(frame_paths),
                        frame_paths=cosmos_frame_paths,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                features[row_index] = self._build_failed_feature(
                    clip_id=clip_id,
                    error=str(exc),
                )

        if self.config.feature.prefetch_batches > 0:
            self._apply_clip_inputs_with_prefetch(clip_inputs, features)
        else:
            self._apply_clip_inputs(clip_inputs, features)

        self.features.extend(feature for feature in features if feature is not None)

    def _apply_clip_inputs(
        self,
        clip_inputs: list[CosmosClipInput],
        features: list[CosmosFeature | None],
    ) -> None:
        forward_batch_size = int(self.config.feature.forward_batch_size)
        for start in range(0, len(clip_inputs), forward_batch_size):
            chunk = clip_inputs[start : start + forward_batch_size]
            self._embed_clip_input_chunk(chunk, features)

    def _apply_clip_inputs_with_prefetch(
        self,
        clip_inputs: list[CosmosClipInput],
        features: list[CosmosFeature | None],
    ) -> None:
        if not clip_inputs:
            return

        embedding_model = self._get_embedding_model()
        forward_batch_size = int(self.config.feature.forward_batch_size)
        frame_batch_size = forward_batch_size * _COSMOS_FRAME_COUNT
        flat_frame_paths = [
            frame_path
            for item in clip_inputs
            for frame_path in item.frame_paths
        ]
        frame_batches = iter(
            iter_rgb_frame_tensor_batches(
                flat_frame_paths,
                batch_size=frame_batch_size,
                max_workers=self.config.feature.frame_load_workers,
                prefetch_batches=self.config.feature.prefetch_batches,
            )
        )

        clip_offset = 0
        while clip_offset < len(clip_inputs):
            chunk_size = min(forward_batch_size, len(clip_inputs) - clip_offset)
            chunk = clip_inputs[clip_offset : clip_offset + chunk_size]
            try:
                frame_batch = next(frame_batches)
                rgb_videos = embedding_model.stack_rgb_video_tensors(frame_batch)
                if len(rgb_videos) != len(chunk):
                    raise RuntimeError(
                        f"Cosmos loaded {len(rgb_videos)} videos for "
                        f"{len(chunk)} clips"
                    )
                embeddings = embedding_model.embed_rgb_videos(rgb_videos)
                self._store_successful_features(chunk, embeddings, features)
            except StopIteration as exc:
                raise RuntimeError(
                    "Cosmos frame batch iterator ended before all clips were processed"
                ) from exc
            except Exception as exc:  # noqa: BLE001
                self._embed_clip_input_chunk_fallback(chunk, features, exc)
            clip_offset += len(chunk)

    def _embed_clip_input_chunk(
        self,
        chunk: list[CosmosClipInput],
        features: list[CosmosFeature | None],
    ) -> None:
        frame_batches = [item.frame_paths for item in chunk]
        try:
            embeddings = self._embed_frame_batches(frame_batches)
            self._store_successful_features(chunk, embeddings, features)
        except Exception as exc:  # noqa: BLE001
            self._embed_clip_input_chunk_fallback(chunk, features, exc)

    def _embed_clip_input_chunk_fallback(
        self,
        chunk: list[CosmosClipInput],
        features: list[CosmosFeature | None],
        exc: Exception,
    ) -> None:
        if len(chunk) == 1:
            item = chunk[0]
            features[item.row_index] = self._build_failed_feature(
                clip_id=item.clip_id,
                error=str(exc),
                input_frame_count=item.frame_count,
                selected_frame_count=len(item.frame_paths),
            )
            return
        for item in chunk:
            try:
                embeddings = self._embed_frame_batches([item.frame_paths])
                self._store_successful_features([item], embeddings, features)
            except Exception as item_exc:  # noqa: BLE001
                features[item.row_index] = self._build_failed_feature(
                    clip_id=item.clip_id,
                    error=str(item_exc),
                    input_frame_count=item.frame_count,
                    selected_frame_count=len(item.frame_paths),
                )

    def _store_successful_features(
        self,
        chunk: list[CosmosClipInput],
        embeddings: np.ndarray,
        features: list[CosmosFeature | None],
    ) -> None:
        if len(embeddings) != len(chunk):
            raise ValueError(
                f"Cosmos returned {len(embeddings)} embeddings for "
                f"{len(chunk)} clips"
            )
        for item, embedding in zip(
            chunk,
            embeddings,
            strict=True,
        ):
            normalized = normalize_embedding(embedding)
            self.embedding_dim = max(self.embedding_dim, int(normalized.size))
            features[item.row_index] = CosmosFeature(
                clip_id=item.clip_id,
                ok=1,
                error="",
                input_frame_count=item.frame_count,
                selected_frame_count=len(item.frame_paths),
                embedding=np.ascontiguousarray(normalized, dtype=np.float16),
            )

    def save_features(self, output_path: Path) -> None:
        output_path.mkdir(parents=True, exist_ok=True)
        embedding_features = [
            feature for feature in self.features if feature.has_embedding
        ]
        embedding_dim = (
            self.embedding_dim
            if self.embedding_dim > 0
            else int(embedding_features[0].embedding.size)
            if embedding_features
            else 0
        )
        feature_rows = [
            {
                "clip_id": feature.clip_id,
                "ok": feature.ok,
                "error": feature.error,
                "input_frame_count": feature.input_frame_count,
                "selected_frame_count": feature.selected_frame_count,
                "embedding_count": int(feature.has_embedding),
            }
            for feature in self.features
        ]
        schema = pa.schema(
            [
                ("clip_id", pa.string()),
                ("ok", pa.int64()),
                ("error", pa.string()),
                ("input_frame_count", pa.int64()),
                ("selected_frame_count", pa.int64()),
                ("embedding_count", pa.int64()),
            ]
        )
        pq.write_table(
            pa.Table.from_pylist(feature_rows, schema=schema),
            output_path / "clip_features.parquet",
        )
        embeddings = (
            np.ascontiguousarray(
                np.vstack(
                    [
                        feature.embedding.reshape(1, embedding_dim)
                        for feature in embedding_features
                    ]
                ),
                dtype=np.float16,
            )
            if embedding_features
            else np.empty((0, embedding_dim), dtype=np.float16)
        )
        np.save(output_path / "embeddings.npy", embeddings)
        metadata = {
            "deduplicator": self.deduplicator_name,
            "version": self.config.version,
            "model_name": self.config.feature.model_name,
            "forward_batch_size": self.config.feature.forward_batch_size,
            "frame_load_workers": self.config.feature.frame_load_workers,
            "prefetch_batches": self.config.feature.prefetch_batches,
            "target_frame_count": _COSMOS_FRAME_COUNT,
            "embedding_dtype": "float16",
            "embedding_shape": [int(embeddings.shape[0]), int(embeddings.shape[1])],
            "clip_count": len(self.features),
            "embedding_count": int(len(embeddings)),
        }
        (output_path / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self._release_embedding_model()

    def load_features(self, feature_root: Path) -> None:
        features: list[CosmosFeature] = []
        embedding_dim = 0

        for shard_path in sorted(path for path in feature_root.iterdir() if path.is_dir()):
            feature_path = shard_path / "clip_features.parquet"
            embedding_path = shard_path / "embeddings.npy"
            if not feature_path.is_file():
                continue

            embeddings = np.load(embedding_path, allow_pickle=False)
            if embeddings.ndim != 2:
                raise ValueError(
                    f"{embedding_path} must contain 2D embeddings, got {embeddings.shape}"
                )
            if embeddings.shape[1] > 0:
                embedding_dim = max(embedding_dim, int(embeddings.shape[1]))
            offset = 0
            for row in pq.read_table(feature_path).to_pylist():
                embedding_count = int(row["embedding_count"])
                if embedding_count not in (0, 1):
                    raise ValueError("cosmos embedding_count must be 0 or 1")
                if embedding_count:
                    embedding = np.ascontiguousarray(
                        embeddings[offset],
                        dtype=np.float16,
                    )
                    offset += 1
                else:
                    embedding = np.empty((0,), dtype=np.float16)
                features.append(
                    CosmosFeature(
                        clip_id=str(row["clip_id"]),
                        ok=int(row["ok"]),
                        error=str(row["error"]),
                        input_frame_count=int(row["input_frame_count"]),
                        selected_frame_count=int(row["selected_frame_count"]),
                        embedding=embedding,
                    )
                )
            if offset != len(embeddings):
                raise ValueError(
                    f"{shard_path} has {len(embeddings)} embeddings but "
                    f"clip metadata uses {offset}"
                )

        self.features = features
        self.embedding_dim = embedding_dim
        self._build_index()

    def unit_count(self) -> int:
        return len(self.features)

    def unit_ids(self) -> list[str]:
        return [feature.clip_id for feature in self.features]

    def find_duplicate_pairs(self, start: int, end: int) -> list[dict[str, object]]:
        if self.config.match.min_cosine_similarity is None:
            return []
        if self.embedding_feature_indices is None:
            raise RuntimeError("load_features() must be called before find_duplicate_pairs().")

        start = max(0, int(start))
        end = min(len(self.features), int(end))
        if start >= end:
            return []

        pair_matches = self._find_pairs_from_embeddings(start, end)
        duplicate_pairs: list[dict[str, object]] = []
        for (left, right), detail in sorted(pair_matches.items()):
            duplicate_pairs.append(
                {
                    "deduplicator": self.deduplicator_name,
                    "unit_id": left,
                    "similar_unit_id": right,
                    "detail": detail,
                }
            )
        return duplicate_pairs

    def build_rows(
        self,
        duplicate_pairs: list[dict[str, object]],
    ) -> dict[str, object]:
        if self.embedding_feature_indices is None:
            raise RuntimeError("load_features() must be called before build_rows().")

        pair_matches: dict[tuple[str, str], dict[str, object]] = {}
        for pair_row in tqdm(
            duplicate_pairs,
            desc="cosmos build pair matches",
            unit="pair",
            mininterval=5.0,
        ):
            if str(pair_row["deduplicator"]) != self.deduplicator_name:
                continue
            left = str(pair_row["unit_id"])
            right = str(pair_row["similar_unit_id"])
            if left == right:
                continue
            pair_key = tuple(sorted((left, right)))
            match = (
                parse_json_object(
                    pair_row["detail"],
                    description="cosmos pair detail",
                )
                or {}
            )
            existing = pair_matches.get(pair_key)
            if existing is None or is_better_cosmos_match(match, existing):
                pair_matches[pair_key] = match

        payloads: dict[str, dict[str, object]] = {}
        errors: dict[str, str] = {}
        for feature in tqdm(
            self.features,
            desc="cosmos build payloads",
            unit="clip",
            mininterval=5.0,
        ):
            payloads[feature.clip_id] = {
                "frame_count": feature.input_frame_count,
                "cosmos_frame_count": feature.selected_frame_count,
                "embedding_dim": self.embedding_dim,
                "model_name": self.config.feature.model_name,
            }
            if feature.ok != 1:
                errors[feature.clip_id] = feature.error

        feature_by_clip_id = {feature.clip_id: feature for feature in self.features}
        clip_ids = list(feature_by_clip_id)
        groups = merge_duplicate_pairs(
            clip_ids,
            tqdm(
                pair_matches,
                total=len(pair_matches),
                desc="cosmos merge duplicate pairs",
                unit="pair",
                mininterval=5.0,
            ),
        )
        pair_matches_by_clip: dict[
            str,
            list[tuple[str, dict[str, object]]],
        ] = defaultdict(list)
        for (left, right), match in tqdm(
            pair_matches.items(),
            total=len(pair_matches),
            desc="cosmos index pair matches",
            unit="pair",
            mininterval=5.0,
        ):
            pair_matches_by_clip[left].append((right, match))
            pair_matches_by_clip[right].append((left, match))

        rows: dict[str, dict[str, object]] = {}
        group_count = 0
        matched_count = 0
        for group in tqdm(
            groups,
            desc="cosmos materialize rows",
            unit="group",
            mininterval=5.0,
        ):
            group_id = ""
            if len(group) > 1:
                group_count += 1
                matched_count += len(group)
                group_id = f"cosmos-{group_count:06d}"
            group_features = [feature_by_clip_id[clip_id] for clip_id in group]
            best_clip_id_in_group = self._select_best_clip_id_from_group(
                group_features
            )
            for clip_id in group:
                payload = dict(payloads[clip_id])
                payload.update(
                    {
                        "group_id": group_id,
                        "group_size": len(group),
                        "is_best_clip_in_group": int(
                            clip_id == best_clip_id_in_group
                        ),
                        "best_clip_id_in_group": best_clip_id_in_group,
                        "best_matched_clip": (
                            self._select_best_matched_clip_in_group(
                                pair_matches=pair_matches_by_clip[clip_id],
                            )
                            if pair_matches_by_clip.get(clip_id)
                            else {}
                        ),
                    }
                )
                rows[clip_id] = {
                    "ok": int(clip_id not in errors),
                    "error": errors.get(clip_id, ""),
                    "json": payload,
                }

        return {
            "rows": rows,
            "summary": {
                "pair_count": len(pair_matches),
                "group_count": group_count,
                "matched_count": matched_count,
            },
        }

    def _select_best_clip_id_from_group(
        self,
        group: list[CosmosFeature],
    ) -> str:
        best_feature = min(
            group,
            key=lambda feature: (
                int(feature.ok != 1),
                -int(feature.selected_frame_count),
                -int(feature.input_frame_count),
                feature.clip_id,
            ),
        )
        return best_feature.clip_id

    def _select_best_matched_clip_in_group(
        self,
        *,
        pair_matches: list[tuple[str, dict[str, object]]],
    ) -> dict[str, object]:
        best_clip_id = ""
        best_pair_match: dict[str, object] | None = None
        for other_clip_id, match in pair_matches:
            if best_pair_match is None:
                best_clip_id = other_clip_id
                best_pair_match = match
                continue
            current_key = (float(match["cosine_similarity"]), other_clip_id)
            best_key = (float(best_pair_match["cosine_similarity"]), best_clip_id)
            if current_key > best_key:
                best_clip_id = other_clip_id
                best_pair_match = match

        if best_pair_match is None:
            return {}
        return {
            "clip_id": best_clip_id,
            **best_pair_match,
        }

    def _embed_frame_batches(self, frame_batches: list[list[Path]]) -> np.ndarray:
        return self._get_embedding_model().embed(frame_batches)

    def _get_embedding_model(self) -> CosmosVideoEmbeddingModel:
        if self._embedding_model is None:
            self._embedding_model = CosmosVideoEmbeddingModel(
                self.config.feature.model_name,
                frame_load_workers=self.config.feature.frame_load_workers,
            )
        return self._embedding_model

    def _build_index(self) -> None:
        embedding_arrays = [
            feature.embedding.astype(np.float32)
            for feature in self.features
            if feature.has_embedding
        ]
        if not embedding_arrays:
            self.embeddings = None
            self.embedding_feature_indices = np.empty((0,), dtype=np.int64)
            self.faiss_embedding = None
            self.faiss_embedding_index = None
            return

        self.embeddings = normalize_embedding(np.vstack(embedding_arrays))
        self.embedding_dim = int(self.embeddings.shape[1])
        self.embedding_feature_indices = np.asarray(
            [
                feature_index
                for feature_index, feature in enumerate(self.features)
                if feature.has_embedding
            ],
            dtype=np.int64,
        )
        self.faiss_embedding = build_flat_ip_index(
            self.embeddings,
            use_gpu=self.use_gpu_faiss,
            use_float16=False,
            backend=self.config.match.index_backend,
        )
        self.faiss_embedding_index = self.faiss_embedding.index

    def _release_embedding_model(self) -> None:
        self._embedding_model = None
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _find_pairs_from_embeddings(
        self,
        start: int,
        end: int,
    ) -> dict[tuple[str, str], dict[str, object]]:
        if (
            self.faiss_embedding_index is None
            or self.embeddings is None
            or self.embedding_feature_indices is None
        ):
            return {}

        threshold = self.config.match.min_cosine_similarity
        if threshold is None:
            return {}

        query_embedding_indices: list[int] = []
        query_feature_indices: list[int] = []
        embedding_position_by_feature_index = {
            int(feature_index): embedding_index
            for embedding_index, feature_index in enumerate(self.embedding_feature_indices)
        }
        for feature_index in range(start, end):
            embedding_index = embedding_position_by_feature_index.get(feature_index)
            if embedding_index is None:
                continue
            query_embedding_indices.append(int(embedding_index))
            query_feature_indices.append(feature_index)
        if not query_embedding_indices:
            return {}

        query_embeddings = np.ascontiguousarray(
            self.embeddings[query_embedding_indices],
            dtype=np.float32,
        )
        top_k = min(int(self.config.match.top_k), int(len(self.embeddings)))
        if top_k <= 0:
            return {}
        scores, index_embedding_ids = self.faiss_embedding_index.search(
            query_embeddings,
            top_k,
        )

        pair_matches: dict[tuple[str, str], dict[str, object]] = {}
        for query_index, query_feature_index in enumerate(query_feature_indices):
            query_clip_id = self.features[query_feature_index].clip_id
            for score, index_embedding_id in zip(
                scores[query_index],
                index_embedding_ids[query_index],
                strict=True,
            ):
                index_embedding_id = int(index_embedding_id)
                if index_embedding_id < 0:
                    continue
                index_feature_index = int(
                    self.embedding_feature_indices[index_embedding_id]
                )
                index_clip_id = self.features[index_feature_index].clip_id
                if query_clip_id == index_clip_id:
                    continue
                score = float(score)
                if score < float(threshold):
                    continue
                match = {"cosine_similarity": round(score, 6)}
                pair_key = tuple(sorted((query_clip_id, index_clip_id)))
                existing = pair_matches.get(pair_key)
                if existing is None or is_better_cosmos_match(match, existing):
                    pair_matches[pair_key] = match
        return pair_matches

    def _build_failed_feature(
        self,
        *,
        clip_id: str,
        error: str,
        input_frame_count: int = 0,
        selected_frame_count: int = 0,
    ) -> CosmosFeature:
        return CosmosFeature(
            clip_id=clip_id,
            ok=0,
            error=error,
            input_frame_count=input_frame_count,
            selected_frame_count=selected_frame_count,
            embedding=np.empty((0,), dtype=np.float16),
        )
