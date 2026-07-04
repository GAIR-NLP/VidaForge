from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from vidaforge.common import join_data_dir, parse_json_object

from .base import DeduplicatorBase
from .config import PDQDedupConfig
from .faiss_index import FaissIndex, build_binary_flat_index
from .grouping import merge_duplicate_pairs

_PDQ_HEX_LENGTH = 64
_PDQ_BYTES = 32
_PDQ_BITS = 256


@dataclass(frozen=True, slots=True)
class PDQFeature:
    clip_id: str
    ok: int
    error: str
    input_frame_count: int
    hashes: np.ndarray
    quality_mean: float

    @property
    def hash_count(self) -> int:
        return int(len(self.hashes))


def pdq_hashes_to_array(hashes: list[str]) -> np.ndarray:
    if not hashes:
        return np.empty((0, _PDQ_BYTES), dtype=np.uint8)

    rows: list[np.ndarray] = []
    for hash_hex in hashes:
        if len(hash_hex) != _PDQ_HEX_LENGTH:
            raise ValueError(f"expected 64-char PDQ hex, got {len(hash_hex)}")
        rows.append(np.frombuffer(bytes.fromhex(hash_hex), dtype=np.uint8))
    return np.ascontiguousarray(np.vstack(rows), dtype=np.uint8)


def is_better_pdq_match(
    candidate: dict[str, object],
    current: dict[str, object],
) -> bool:
    def key(match: dict[str, object]) -> tuple[float, float, int]:
        return (
            float(match["similar_frame_ratio"]),
            -float(match["mean_hamming_distance"]),
            int(match["similar_frame_count"]),
        )

    return key(candidate) > key(current)


def compute_pdq_hashes(
    frame_paths: list[Path],
    *,
    min_quality: float,
) -> tuple[list[str], list[float]]:
    import pdqhash

    hashes: list[str] = []
    quality_values: list[float] = []

    for frame_path in frame_paths:
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to read frame: {frame_path}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hash_bits, quality = pdqhash.compute(rgb)
        quality_value = float(quality)
        if quality_value < min_quality:
            continue
        bits = np.asarray(hash_bits, dtype=np.uint8)
        if bits.size != _PDQ_BITS:
            raise ValueError(f"expected 256-bit PDQ hash, got {bits.size} bits")
        hash_hex = np.packbits(bits).tobytes().hex()
        if len(hash_hex) != _PDQ_HEX_LENGTH:
            raise ValueError(f"expected 64-char PDQ hex, got {len(hash_hex)}")
        hashes.append(hash_hex)
        quality_values.append(quality_value)

    return hashes, quality_values


class PDQDeduplicator(DeduplicatorBase[PDQDedupConfig]):
    deduplicator_name: ClassVar[str] = "pdq"
    config_type: ClassVar[type[PDQDedupConfig]] = PDQDedupConfig

    def __init__(
        self,
        config: PDQDedupConfig,
        *,
        use_gpu_faiss: bool = False,
    ) -> None:
        super().__init__(config, use_gpu_faiss=use_gpu_faiss)
        self.features: list[PDQFeature] = []
        self.faiss_hash: FaissIndex | None = None
        self.faiss_hash_index: Any | None = None
        self.faiss_hash_clip_indices: np.ndarray | None = None

    def apply(self, row: dict[str, object]) -> None:
        clip_id = str(row["clip_id"])
        ok = 1
        error = ""
        input_frame_count = 0
        hashes = np.empty((0, _PDQ_BYTES), dtype=np.uint8)
        quality_mean = 0.0

        try:
            frame_json = parse_json_object(row["frame_json"], description="frame_json")
            if frame_json is None:
                raise ValueError("frame_json is missing")
            frame_paths_value = frame_json["frame_paths"]
            if not isinstance(frame_paths_value, list):
                raise ValueError("frame_json.frame_paths must be a list")
            frame_paths = [join_data_dir(str(path)) for path in frame_paths_value]
            input_frame_count = len(frame_paths)
            hash_values, quality_values = compute_pdq_hashes(
                frame_paths,
                min_quality=self.config.feature.min_quality,
            )
            hashes = pdq_hashes_to_array(hash_values)
            quality_mean = (
                round(float(np.mean(quality_values)), 6)
                if quality_values
                else 0.0
            )
        except Exception as exc:  # noqa: BLE001
            ok = 0
            error = str(exc)

        self.features.append(
            PDQFeature(
                clip_id=clip_id,
                ok=int(ok),
                error=error,
                input_frame_count=input_frame_count,
                hashes=hashes,
                quality_mean=quality_mean,
            )
        )

    def save_features(self, output_path: Path) -> None:
        output_path.mkdir(parents=True, exist_ok=True)
        feature_rows = [
            {
                "clip_id": feature.clip_id,
                "ok": feature.ok,
                "error": feature.error,
                "input_frame_count": feature.input_frame_count,
                "hash_count": feature.hash_count,
                "quality_mean": feature.quality_mean,
            }
            for feature in self.features
        ]
        schema = pa.schema(
            [
                ("clip_id", pa.string()),
                ("ok", pa.int64()),
                ("error", pa.string()),
                ("input_frame_count", pa.int64()),
                ("hash_count", pa.int64()),
                ("quality_mean", pa.float64()),
            ]
        )
        table = pa.Table.from_pylist(feature_rows, schema=schema)
        pq.write_table(table, output_path / "clip_features.parquet")
        hashes = (
            np.ascontiguousarray(
                np.vstack([feature.hashes for feature in self.features if feature.hash_count]),
                dtype=np.uint8,
            )
            if any(feature.hash_count for feature in self.features)
            else np.empty((0, _PDQ_BYTES), dtype=np.uint8)
        )
        np.save(output_path / "hashes.npy", hashes)
        metadata = {
            "deduplicator": self.deduplicator_name,
            "version": self.config.version,
            "hash_dtype": "uint8",
            "hash_shape": [int(len(hashes)), _PDQ_BYTES],
            "clip_count": len(self.features),
            "hash_count": int(len(hashes)),
        }
        (output_path / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def load_features(self, feature_root: Path) -> None:
        features: list[PDQFeature] = []

        for shard_path in sorted(path for path in feature_root.iterdir() if path.is_dir()):
            feature_path = shard_path / "clip_features.parquet"
            hash_path = shard_path / "hashes.npy"
            if not feature_path.is_file():
                continue

            hashes = np.ascontiguousarray(
                np.load(hash_path).reshape(-1, _PDQ_BYTES),
                dtype=np.uint8,
            )
            offset = 0
            for row in pq.read_table(feature_path).to_pylist():
                hash_count = int(row["hash_count"])
                features.append(
                    PDQFeature(
                        clip_id=str(row["clip_id"]),
                        ok=int(row["ok"]),
                        error=str(row["error"]),
                        input_frame_count=int(row["input_frame_count"]),
                        hashes=np.ascontiguousarray(
                            hashes[offset : offset + hash_count],
                            dtype=np.uint8,
                        ),
                        quality_mean=float(row["quality_mean"]),
                    )
                )
                offset += hash_count
            if offset != len(hashes):
                raise ValueError(
                    f"{shard_path} has {len(hashes)} hashes but clip metadata uses {offset}"
                )

        self.features = features
        self._build_index()

    def unit_count(self) -> int:
        return len(self.features)

    def unit_ids(self) -> list[str]:
        return [feature.clip_id for feature in self.features]

    def find_duplicate_pairs(self, start: int, end: int) -> list[dict[str, object]]:
        if self.faiss_hash_clip_indices is None:
            raise RuntimeError("load_features() must be called before find_duplicate_pairs().")
        start = max(0, int(start))
        end = min(len(self.features), int(end))
        if start >= end:
            return []

        query_features = self.features[start:end]
        pair_match_results = self._find_pairs_from_hashes(query_features)
        duplicate_pairs: list[dict[str, object]] = []
        for (left, right), detail in sorted(pair_match_results.items()):
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
        if self.faiss_hash_clip_indices is None:
            raise RuntimeError("load_features() must be called before build_rows().")

        pair_matches: dict[tuple[str, str], dict[str, object]] = {}
        # The same clip pair can be found from both query directions or ranges.
        # Keep the strongest evidence for each unordered pair.
        for pair_row in tqdm(
            duplicate_pairs,
            desc="pdq build pair matches",
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
                    description="pdq pair detail",
                )
                or {}
            )
            existing = pair_matches.get(pair_key)
            if existing is None or is_better_pdq_match(match, existing):
                pair_matches[pair_key] = match

        payloads: dict[str, dict[str, object]] = {}
        errors: dict[str, str] = {}
        for feature in tqdm(
            self.features,
            desc="pdq build payloads",
            unit="clip",
            mininterval=5.0,
        ):
            payloads[feature.clip_id] = {
                "frame_count": feature.input_frame_count,
                "pdq_frame_count": feature.hash_count,
                "pdq_quality_mean": feature.quality_mean,
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
                desc="pdq merge duplicate pairs",
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
            desc="pdq index pair matches",
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
            desc="pdq materialize rows",
            unit="group",
            mininterval=5.0,
        ):
            group_id = ""
            if len(group) > 1:
                group_count += 1
                matched_count += len(group)
                group_id = f"pdq-{group_count:06d}"
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
        group: list[PDQFeature],
    ) -> str:
        best_feature = min(
            group,
            key=lambda feature: (
                int(feature.ok != 1),
                -float(feature.quality_mean),
                -int(feature.hash_count),
                -int(feature.input_frame_count),
                feature.clip_id,
            ),
        )
        return best_feature.clip_id

    def _build_index(self) -> None:
        hash_arrays = [feature.hashes for feature in self.features if feature.hash_count]
        if not hash_arrays:
            self.faiss_hash = None
            self.faiss_hash_index = None
            self.faiss_hash_clip_indices = np.empty((0,), dtype=np.int64)
            return

        hashes = np.ascontiguousarray(np.vstack(hash_arrays), dtype=np.uint8)
        self.faiss_hash_clip_indices = np.concatenate(
            [
                np.full((feature.hash_count,), clip_index, dtype=np.int64)
                for clip_index, feature in enumerate(self.features)
                if feature.hash_count
            ]
        )
        self.faiss_hash = build_binary_flat_index(
            hashes,
            bits=_PDQ_BITS,
            use_gpu=False,
        )
        self.faiss_hash_index = self.faiss_hash.index

    def _find_pairs_from_hashes(
        self,
        query_features: list[PDQFeature],
    ) -> dict[tuple[str, str], dict[str, object]]:
        if self.faiss_hash_index is None or self.faiss_hash_clip_indices is None:
            return {}

        query_hash_list = [
            feature.hashes for feature in query_features if feature.hash_count
        ]
        if not query_hash_list:
            return {}

        query_clip_ids = [feature.clip_id for feature in query_features]
        feature_clip_ids = [feature.clip_id for feature in self.features]
        query_hashes = np.ascontiguousarray(np.vstack(query_hash_list), dtype=np.uint8)
        query_hash_clip_indices = np.concatenate(
            [
                np.full((feature.hash_count,), clip_index, dtype=np.int64)
                for clip_index, feature in enumerate(query_features)
                if feature.hash_count
            ]
        )

        top_k = min(int(self.config.match.top_k), int(self.faiss_hash_index.ntotal))
        if top_k <= 0:
            return {}
        distances, index_hash_ids = self.faiss_hash_index.search(query_hashes, top_k)
        directed_distances: dict[tuple[int, int], dict[int, int]] = defaultdict(dict)
        query_hash_counts = np.bincount(
            query_hash_clip_indices,
            minlength=len(query_clip_ids),
        )
        hamming_threshold = int(self.config.match.hamming_distance_threshold)

        for query_hash_index, query_clip_index in enumerate(query_hash_clip_indices):
            query_clip_index = int(query_clip_index)
            query_clip_id = query_clip_ids[query_clip_index]
            for distance, index_hash_index in zip(
                distances[query_hash_index],
                index_hash_ids[query_hash_index],
                strict=True,
            ):
                index_hash_index = int(index_hash_index)
                if index_hash_index < 0:
                    continue
                distance = int(distance)
                if distance > hamming_threshold:
                    continue
                index_clip_index = int(self.faiss_hash_clip_indices[index_hash_index])
                index_clip_id = feature_clip_ids[index_clip_index]
                if query_clip_id == index_clip_id:
                    continue
                pair_key = (query_clip_index, index_clip_index)
                current_distance = directed_distances[pair_key].get(query_hash_index)
                if current_distance is None or distance < current_distance:
                    directed_distances[pair_key][query_hash_index] = distance

        pair_matches: dict[tuple[str, str], dict[str, object]] = {}
        for (query_clip_index, index_clip_index), hash_distances in sorted(
            directed_distances.items()
        ):
            query_clip_id = query_clip_ids[query_clip_index]
            index_clip_id = feature_clip_ids[index_clip_index]
            query_hash_count = int(query_hash_counts[query_clip_index])
            if query_hash_count <= 0:
                continue
            similar_hash_count = len(hash_distances)
            similar_frame_ratio = float(similar_hash_count / query_hash_count)
            if similar_frame_ratio < self.config.match.min_similar_frame_ratio:
                continue
            match = {
                "similar_frame_count": similar_hash_count,
                "similar_frame_ratio": round(similar_frame_ratio, 6),
                "mean_hamming_distance": round(
                    float(np.mean(list(hash_distances.values()))),
                    6,
                ),
            }
            pair_key = tuple(sorted((query_clip_id, index_clip_id)))
            existing = pair_matches.get(pair_key)
            if existing is None or is_better_pdq_match(match, existing):
                pair_matches[pair_key] = match
        return pair_matches

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
            current_key = (
                float(match["similar_frame_ratio"]),
                -float(match["mean_hamming_distance"]),
                other_clip_id,
            )
            best_key = (
                float(best_pair_match["similar_frame_ratio"]),
                -float(best_pair_match["mean_hamming_distance"]),
                best_clip_id,
            )
            if current_key > best_key:
                best_clip_id = other_clip_id
                best_pair_match = match

        if best_pair_match is None:
            return {}
        return {
            "clip_id": best_clip_id,
            **best_pair_match,
        }
