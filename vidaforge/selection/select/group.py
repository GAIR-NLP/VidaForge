"""Select retained clips within candidate groups after quality filtering."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


@dataclass
class GroupSelection:
    keep_ids: set[str]
    details: dict[str, dict[str, object]] = field(default_factory=dict)


class GroupSelector:
    """Keep the requested number of clips from an already ranked group."""

    required_score_fields: tuple[str, ...] = ()

    def select(self, ranked_ids: list[str], keep_count: int) -> GroupSelection:
        return GroupSelection(set(ranked_ids[:keep_count]))


class CosmosGroupSelector(GroupSelector):
    """Reject only clips directly similar to the fixed initial selection."""

    required_score_fields = (
        "aesthetic_score", "text_score", "optical_score", "motion_score"
    )

    def __init__(self, embeddings: dict[str, np.ndarray], threshold: float) -> None:
        self.embeddings = embeddings
        self.threshold = threshold

    def _embedding(self, clip_id: str) -> np.ndarray:
        try:
            embedding = np.asarray(self.embeddings[clip_id], dtype=np.float32)
        except KeyError as exc:
            raise ValueError(f"missing Cosmos embedding for {clip_id}") from exc
        norm = np.linalg.norm(embedding)
        if not np.isfinite(embedding).all() or not np.isfinite(norm) or norm <= 0:
            raise ValueError(f"invalid Cosmos embedding for {clip_id}")
        return embedding / norm

    def select(self, ranked_ids: list[str], keep_count: int) -> GroupSelection:
        initial_ids = ranked_ids[:keep_count]
        result = GroupSelection(set(initial_ids))
        if keep_count == len(ranked_ids):
            return result
        initial_embeddings = [self._embedding(clip_id) for clip_id in initial_ids]
        for clip_id in ranked_ids[keep_count:]:
            embedding = self._embedding(clip_id)
            for initial_id, initial_embedding in zip(initial_ids, initial_embeddings):
                similarity = float(np.dot(embedding, initial_embedding))
                if similarity >= self.threshold:
                    result.details[clip_id] = {
                        "matched_kept_clip_id": initial_id,
                        "cosine_similarity": round(similarity, 6),
                    }
                    break
            else:
                result.keep_ids.add(clip_id)
        return result


def build_group_selectors(
    dedup_config: dict[str, dict[str, object]], input_path: Path
) -> dict[str, GroupSelector]:
    selectors = {}
    for method in dedup_config:
        if method == "dedup_ok":
            continue
        if method == "cosmos":
            embeddings, threshold = load_cosmos_similarity_input(input_path)
            selectors[method] = CosmosGroupSelector(embeddings, threshold)
        else:
            selectors[method] = GroupSelector()
    return selectors


def load_cosmos_similarity_input(input_path: Path) -> tuple[dict[str, np.ndarray], float]:
    summary_path = input_path / "summary.json"
    if not summary_path.is_file():
        raise ValueError(f"Cosmos select requires dedup summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    try:
        threshold = summary["deduplicator"]["cosmos"]["match"]["min_cosine_similarity"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Cosmos dedup summary is missing its similarity threshold") from exc
    if threshold is None:
        raise ValueError("Cosmos select requires a configured min_cosine_similarity")

    feature_root = input_path / "features" / "cosmos"
    if not feature_root.is_dir():
        raise ValueError(f"Cosmos select requires saved embeddings: {feature_root}")

    embeddings_by_clip: dict[str, np.ndarray] = {}
    for shard_path in sorted(path for path in feature_root.iterdir() if path.is_dir()):
        feature_path = shard_path / "clip_features.parquet"
        embedding_path = shard_path / "embeddings.npy"
        if not feature_path.is_file() or not embedding_path.is_file():
            raise ValueError(f"Cosmos feature shard is incomplete: {shard_path}")
        embeddings = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
        if embeddings.ndim != 2:
            raise ValueError(f"Cosmos embeddings must be 2D: {embedding_path}")
        offset = 0
        for row in pq.read_table(feature_path, columns=["clip_id", "embedding_count"]).to_pylist():
            clip_id = str(row["clip_id"])
            if clip_id in embeddings_by_clip:
                raise ValueError(f"duplicate Cosmos feature clip_id: {clip_id}")
            count = int(row["embedding_count"])
            if count not in (0, 1):
                raise ValueError(f"invalid Cosmos embedding_count for {clip_id}: {count}")
            if count:
                if offset >= len(embeddings):
                    raise ValueError(f"Cosmos feature shard has too few embeddings: {shard_path}")
                embeddings_by_clip[clip_id] = embeddings[offset]
                offset += 1
        if offset != len(embeddings):
            raise ValueError(f"Cosmos feature shard has unused embeddings: {shard_path}")
    return embeddings_by_clip, float(threshold)
