from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from vidaforge.common import utc_now_iso
from vidaforge.index import iter_parquet, write_parquet_shards
from vidaforge.packaging.automodel.output import write_metadata_files


_CANDIDATE_COLUMNS = [
    "clip_id",
    "video_id",
    "select_pass",
    "automodel_ok",
    "automodel_cache_file",
    "automodel_bucket_width",
    "automodel_bucket_height",
    "automodel_bucket_frame_count",
    "automodel_latent_shape",
]


@dataclass(frozen=True, slots=True)
class SplitComponent:
    select_pass: int
    count: int


@dataclass(frozen=True, slots=True)
class SplitSpec:
    name: str
    components: tuple[SplitComponent, ...]


@dataclass(frozen=True, slots=True)
class CandidateRow:
    clip_id: str
    video_id: str
    select_pass: int


def parse_component_spec(value: str) -> SplitComponent:
    """Parse ``select_pass=1,count=10000`` style component specs."""
    fields: dict[str, str] = {}
    for part in value.split(","):
        key, separator, raw_item_value = part.partition("=")
        if not separator:
            raise ValueError(f"invalid component item {part!r}: expected key=value")
        key = key.strip()
        item_value = raw_item_value.strip()
        if not key or not item_value:
            raise ValueError(f"invalid component item {part!r}: empty key or value")
        if key in fields:
            raise ValueError(f"duplicate component key: {key}")
        fields[key] = item_value

    allowed_keys = {"select_pass", "count"}
    unknown_keys = sorted(set(fields) - allowed_keys)
    if unknown_keys:
        raise ValueError(f"unknown component keys: {unknown_keys}")
    if set(fields) != allowed_keys:
        raise ValueError("component must contain exactly select_pass and count")

    try:
        select_pass = int(fields["select_pass"])
    except ValueError as exc:
        raise ValueError(f"select_pass must be 0 or 1: {value!r}") from exc
    if select_pass not in {0, 1}:
        raise ValueError(f"select_pass must be 0 or 1: {value!r}")

    try:
        count = int(fields["count"])
    except ValueError as exc:
        raise ValueError(f"count must be a positive integer: {value!r}") from exc
    if count <= 0:
        raise ValueError(f"count must be a positive integer: {value!r}")

    return SplitComponent(select_pass=select_pass, count=count)


def create_automodel_cache_splits(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    train_spec: SplitSpec | None = None,
    val_spec: SplitSpec | None = None,
    exclude_video_ids_paths: Iterable[str | Path] = (),
    seed: int = 42,
    metadata_shard_size: int = 10_000,
    parquet_size: int = 500_000,
    overwrite: bool = False,
    check_meta: bool = False,
) -> dict[str, object]:
    if train_spec is None and val_spec is None:
        raise ValueError("at least one of train_spec or val_spec is required")
    _validate_split_specs(train_spec=train_spec, val_spec=val_spec)
    if metadata_shard_size <= 0:
        raise ValueError("metadata_shard_size must be > 0")
    if parquet_size <= 0:
        raise ValueError("parquet_size must be > 0")

    input_path = Path(input_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    _validate_requested_outputs(
        output_path=output_path,
        specs=[spec for spec in (train_spec, val_spec) if spec is not None],
        overwrite=overwrite,
    )

    started_at = utc_now_iso()
    excluded_video_ids = _read_video_id_files(exclude_video_ids_paths)
    candidates = _read_candidates(input_path, excluded_video_ids=excluded_video_ids)

    selected_by_split: dict[str, list[CandidateRow]] = {}
    component_summaries_by_split: dict[str, list[dict[str, object]]] = {}

    rng = random.Random(seed)
    consumed_clip_ids: set[str] = set()

    if val_spec is not None:
        selected, component_summaries = _sample_split(
            spec=val_spec,
            candidates=candidates,
            rng=rng,
            consumed_clip_ids=set(),
            forbidden_video_ids=set(),
        )
        selected_by_split[val_spec.name] = selected
        component_summaries_by_split[val_spec.name] = component_summaries
        consumed_clip_ids.update(candidate.clip_id for candidate in selected)

    val_video_ids = {
        candidate.video_id
        for split_name, selected in selected_by_split.items()
        if val_spec is not None and split_name == val_spec.name
        for candidate in selected
    }

    if train_spec is not None:
        selected, component_summaries = _sample_split(
            spec=train_spec,
            candidates=candidates,
            rng=rng,
            consumed_clip_ids=consumed_clip_ids,
            forbidden_video_ids=val_video_ids,
        )
        selected_by_split[train_spec.name] = selected
        component_summaries_by_split[train_spec.name] = component_summaries

    selected_clip_ids = {
        candidate.clip_id
        for selected in selected_by_split.values()
        for candidate in selected
    }
    full_rows_by_clip_id = _load_selected_rows(
        input_path=input_path,
        selected_clip_ids=selected_clip_ids,
    )
    missing_clip_ids = sorted(selected_clip_ids - set(full_rows_by_clip_id))
    if missing_clip_ids:
        raise ValueError(
            "selected clip_id values disappeared while loading full rows: "
            f"{missing_clip_ids[:10]}"
        )

    split_summaries: list[dict[str, object]] = []
    for split_name, selected in selected_by_split.items():
        rows = [full_rows_by_clip_id[candidate.clip_id] for candidate in selected]
        if check_meta:
            _validate_selected_meta_files(split_name=split_name, rows=rows)
        split_output_path = output_path / split_name
        split_summary = _write_split_output(
            input_path=input_path,
            output_path=split_output_path,
            rows=rows,
            components=component_summaries_by_split[split_name],
            seed=seed,
            metadata_shard_size=metadata_shard_size,
            parquet_size=parquet_size,
            overwrite=overwrite,
            excluded_video_id_count=len(excluded_video_ids),
            validation_video_id_count=len(val_video_ids),
        )
        split_summaries.append(split_summary)

    summary = {
        "created_at": utc_now_iso(),
        "started_at": started_at,
        "input_dir": str(input_path),
        "output_dir": str(output_path),
        "seed": seed,
        "metadata_shard_size": metadata_shard_size,
        "parquet_size": parquet_size,
        "overwrite": overwrite,
        "check_meta": check_meta,
        "external_excluded_video_id_count": len(excluded_video_ids),
        "candidate_count": len(candidates),
        "splits": split_summaries,
    }
    return summary


def _read_candidates(
    input_path: Path,
    *,
    excluded_video_ids: set[str],
) -> list[CandidateRow]:
    candidates_by_clip_id: dict[str, CandidateRow] = {}
    for row in iter_parquet(input_path, unit="clip", columns=_CANDIDATE_COLUMNS):
        if not _is_complete_manifest_row(row):
            continue

        select_pass = int(row["select_pass"])
        if select_pass not in {0, 1}:
            continue

        clip_id = str(row.get("clip_id", "")).strip()
        video_id = str(row.get("video_id", "")).strip()
        if not clip_id:
            raise ValueError("input cache contains a row with empty clip_id")
        if not video_id:
            raise ValueError(f"input cache contains empty video_id for clip_id={clip_id}")
        if video_id in excluded_video_ids:
            continue

        candidates_by_clip_id[clip_id] = CandidateRow(
            clip_id=clip_id,
            video_id=video_id,
            select_pass=select_pass,
        )
    candidates = list(candidates_by_clip_id.values())
    if not candidates:
        raise ValueError(f"no usable AutoModel rows found in input cache: {input_path}")
    return candidates


def _validate_split_specs(
    *,
    train_spec: SplitSpec | None,
    val_spec: SplitSpec | None,
) -> None:
    specs = [spec for spec in (train_spec, val_spec) if spec is not None]
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"split names must be unique: {names}")
    for spec in specs:
        _validate_split_name(spec.name)
        if not spec.components:
            raise ValueError(f"split={spec.name!r} must contain at least one component")


def _validate_split_name(name: str) -> None:
    if not name.strip():
        raise ValueError("split name must not be empty")
    path = Path(name)
    if path.is_absolute() or len(path.parts) != 1 or path.parts[0] in {".", ".."}:
        raise ValueError(f"split name must be one path component: {name!r}")


def _validate_requested_outputs(
    *,
    output_path: Path,
    specs: list[SplitSpec],
    overwrite: bool,
) -> None:
    if overwrite:
        return
    existing = [str(output_path / spec.name) for spec in specs if (output_path / spec.name).exists()]
    if existing:
        raise FileExistsError(f"split output already exists: {existing}")


def _sample_split(
    *,
    spec: SplitSpec,
    candidates: list[CandidateRow],
    rng: random.Random,
    consumed_clip_ids: set[str],
    forbidden_video_ids: set[str],
) -> tuple[list[CandidateRow], list[dict[str, object]]]:
    selected: list[CandidateRow] = []
    local_consumed_clip_ids = set(consumed_clip_ids)
    component_summaries: list[dict[str, object]] = []

    for component_index, component in enumerate(spec.components):
        available = [
            candidate
            for candidate in candidates
            if candidate.select_pass == component.select_pass
            and candidate.clip_id not in local_consumed_clip_ids
            and candidate.video_id not in forbidden_video_ids
        ]
        available_count = len(available)
        if available_count < component.count:
            raise ValueError(
                f"not enough rows for split={spec.name!r} component "
                f"select_pass={component.select_pass},count={component.count}: "
                f"available={available_count} after filters and excluded video_ids"
            )

        sampled = rng.sample(available, component.count)
        selected.extend(sampled)
        local_consumed_clip_ids.update(candidate.clip_id for candidate in sampled)
        component_summaries.append(
            {
                "component_index": component_index,
                "select_pass": component.select_pass,
                "requested_count": component.count,
                "available_count": available_count,
                "selected_count": len(sampled),
            }
        )

    return selected, component_summaries


def _load_selected_rows(
    *,
    input_path: Path,
    selected_clip_ids: set[str],
) -> dict[str, dict[str, object]]:
    rows_by_clip_id: dict[str, dict[str, object]] = {}
    if not selected_clip_ids:
        return rows_by_clip_id
    for row in iter_parquet(input_path, unit="clip"):
        clip_id = str(row.get("clip_id", "")).strip()
        if clip_id in selected_clip_ids and _is_complete_manifest_row(row):
            rows_by_clip_id[clip_id] = dict(row)
    return rows_by_clip_id


def _is_complete_manifest_row(row: dict[str, object]) -> bool:
    try:
        if int(row["automodel_ok"]) != 1:
            return False
        if not str(row.get("automodel_cache_file", "")).strip():
            return False
        if int(row.get("automodel_bucket_width", 0) or 0) <= 0:
            return False
        if int(row.get("automodel_bucket_height", 0) or 0) <= 0:
            return False
        if int(row.get("automodel_bucket_frame_count", 0) or 0) <= 0:
            return False
    except (KeyError, TypeError, ValueError):
        return False

    return _has_non_empty_latent_shape(row.get("automodel_latent_shape"))


def _has_non_empty_latent_shape(value: object) -> bool:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return False
    return isinstance(value, list | tuple) and bool(value)


def _write_split_output(
    *,
    input_path: Path,
    output_path: Path,
    rows: list[dict[str, object]],
    components: list[dict[str, object]],
    seed: int,
    metadata_shard_size: int,
    parquet_size: int,
    overwrite: bool,
    excluded_video_id_count: int,
    validation_video_id_count: int,
) -> dict[str, object]:
    temp_path = _prepare_temp_output_path(output_path, overwrite=overwrite)
    try:
        parquet_summary = write_parquet_shards(
            rows,
            temp_path,
            unit="clip",
            parquet_size=parquet_size,
            write_summary=False,
        )
        metadata_summary = write_metadata_files(
            output_path=temp_path,
            rows=rows,
            metadata_shard_size=metadata_shard_size,
        )
        metadata_summary_for_output = dict(metadata_summary)
        metadata_summary_for_output["metadata_path"] = str(output_path / "metadata.json")
        clip_ids = [str(row["clip_id"]) for row in rows]
        video_ids = sorted({str(row["video_id"]) for row in rows})
        (temp_path / "clip_ids.txt").write_text(
            "".join(f"{clip_id}\n" for clip_id in clip_ids),
            encoding="utf-8",
        )
        (temp_path / "video_ids.txt").write_text(
            "".join(f"{video_id}\n" for video_id in video_ids),
            encoding="utf-8",
        )

        summary = {
            "created_at": utc_now_iso(),
            "input_dir": str(input_path),
            "output_path": str(output_path),
            "seed": seed,
            "row_count": len(rows),
            "unique_video_id_count": len(video_ids),
            "metadata_shard_size": metadata_shard_size,
            "parquet_size": parquet_size,
            "external_excluded_video_id_count": excluded_video_id_count,
            "validation_video_id_count": validation_video_id_count,
            "components": components,
            "select_pass_distribution": _value_distribution(
                int(row["select_pass"]) for row in rows
            ),
            "bucket_frame_count_distribution": _value_distribution(
                int(row["automodel_bucket_frame_count"]) for row in rows
            ),
            "bucket_resolution_distribution": _value_distribution(
                f"{int(row['automodel_bucket_width'])}x"
                f"{int(row['automodel_bucket_height'])}"
                for row in rows
            ),
            "parquet": {
                "shard_count": parquet_summary["shard_count"],
                "total_rows": parquet_summary["total_rows"],
                "shards": parquet_summary["shards"],
            },
            "metadata": metadata_summary_for_output,
        }
        _write_json(temp_path / "summary.json", summary)
        _commit_temp_output_path(temp_path, output_path, overwrite=overwrite)
    except Exception:
        shutil.rmtree(temp_path, ignore_errors=True)
        raise

    committed_summary = dict(summary)
    committed_summary["output_path"] = str(output_path)
    return committed_summary


def _prepare_temp_output_path(output_path: Path, *, overwrite: bool) -> Path:
    output_path = output_path.expanduser().resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"split output already exists: {output_path}")
    temp_path = output_path.parent / f".{output_path.name}.tmp-{os.getpid()}"
    shutil.rmtree(temp_path, ignore_errors=True)
    temp_path.mkdir(parents=True)
    return temp_path


def _commit_temp_output_path(
    temp_path: Path,
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    output_path = output_path.expanduser().resolve()
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"split output already exists: {output_path}")
        shutil.rmtree(output_path)
    temp_path.rename(output_path)


def _validate_selected_meta_files(
    *,
    split_name: str,
    rows: list[dict[str, object]],
) -> None:
    missing: list[str] = []
    for row in rows:
        path = Path(str(row["automodel_cache_file"])).expanduser().resolve()
        try:
            if not path.is_file() or path.stat().st_size <= 0:
                missing.append(str(path))
        except OSError:
            missing.append(str(path))
        if len(missing) >= 10:
            break
    if missing:
        raise FileNotFoundError(
            f"split={split_name!r} selected missing or empty .meta files: {missing}"
        )


def _read_video_id_files(paths: Iterable[str | Path]) -> set[str]:
    video_ids: set[str] = set()
    for path_like in paths:
        path = Path(path_like).expanduser().resolve()
        for line in path.read_text(encoding="utf-8").splitlines():
            video_id = line.strip()
            if video_id:
                video_ids.add(video_id)
    return video_ids


def _value_distribution(values: Iterable[object]) -> dict[str, dict[str, float | int]]:
    counts: dict[str, int] = {}
    total = 0
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
        total += 1
    if total <= 0:
        return {}
    return {
        key: {
            "count": count,
            "ratio": round(count / total, 6),
        }
        for key, count in sorted(counts.items())
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_split_spec(
    *,
    name: str | None,
    components: list[str],
    kind: str,
) -> SplitSpec | None:
    if not components:
        if name:
            raise ValueError(f"--{kind}-name requires at least one --{kind}-component")
        return None
    if not name:
        raise ValueError(f"--{kind}-component requires --{kind}-name")
    return SplitSpec(
        name=name,
        components=tuple(parse_component_spec(component) for component in components),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create lightweight train/val AutoModel dataset manifests from a "
            "packed Stage 5 AutoModel output directory."
        )
    )
    parser.add_argument(
        "--input-dir",
        "--input-cache",
        dest="input_dir",
        type=Path,
        required=True,
        help="Packed Stage 5 AutoModel output directory containing clip-*.parquet.",
    )
    parser.add_argument(
        "--output-dir",
        "--output-root",
        dest="output_dir",
        type=Path,
        required=True,
        help="Directory where split cache directories will be created.",
    )
    parser.add_argument("--train-name", default=None)
    parser.add_argument(
        "--train-component",
        action="append",
        default=[],
        help='Train component, for example "select_pass=1,count=200000".',
    )
    parser.add_argument("--val-name", default=None)
    parser.add_argument(
        "--val-component",
        action="append",
        default=[],
        help='Validation component, for example "select_pass=1,count=10000".',
    )
    parser.add_argument(
        "--exclude-video-ids",
        action="append",
        default=[],
        type=Path,
        help=(
            "Text file with one video_id per line. Rows from these videos are "
            "excluded before sampling."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata-shard-size", type=int, default=10_000)
    parser.add_argument("--parquet-size", type=int, default=500_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--check-meta",
        action="store_true",
        help="Stat selected .meta files before writing outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_spec = _build_split_spec(
        name=args.train_name,
        components=args.train_component,
        kind="train",
    )
    val_spec = _build_split_spec(
        name=args.val_name,
        components=args.val_component,
        kind="val",
    )
    summary = create_automodel_cache_splits(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_spec=train_spec,
        val_spec=val_spec,
        exclude_video_ids_paths=args.exclude_video_ids,
        seed=args.seed,
        metadata_shard_size=args.metadata_shard_size,
        parquet_size=args.parquet_size,
        overwrite=args.overwrite,
        check_meta=args.check_meta,
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
