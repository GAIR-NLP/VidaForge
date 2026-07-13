from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from vidaforge.common import join_data_dir, utc_now_iso
from vidaforge.index import iter_parquet, write_parquet_shards


def create_vjepa2_manifest_from_automodel_split(
    *,
    automodel_dir: str | Path | None,
    output_dir: str | Path,
    clip_ids_path: str | Path | None = None,
    metadata_dir: str | Path | None = None,
    manifest_name: str = "train.csv",
    label: int = 0,
    parquet_size: int = 500_000,
    overwrite: bool = False,
    check_files: bool = False,
) -> dict[str, object]:
    if automodel_dir is None and (clip_ids_path is None or metadata_dir is None):
        raise ValueError("set --automodel-dir, or set both --clip-ids and --metadata-dir")
    if Path(manifest_name).name != manifest_name or not manifest_name.endswith(".csv"):
        raise ValueError("manifest_name must be a .csv filename")
    if parquet_size <= 0:
        raise ValueError("parquet_size must be > 0")

    automodel_path = Path(automodel_dir).expanduser().resolve() if automodel_dir is not None else None
    resolved_clip_ids_path = (
        Path(clip_ids_path).expanduser().resolve()
        if clip_ids_path is not None
        else automodel_path / "clip_ids.txt"  # type: ignore[operator]
    )
    resolved_metadata_dir = (
        Path(metadata_dir).expanduser().resolve()
        if metadata_dir is not None
        else automodel_path  # type: ignore[assignment]
    )
    output_path = Path(output_dir).expanduser().resolve()

    clip_ids = read_clip_ids(resolved_clip_ids_path)
    rows_by_clip_id = load_rows_for_clip_ids(resolved_metadata_dir, set(clip_ids))
    missing_clip_ids = [clip_id for clip_id in clip_ids if clip_id not in rows_by_clip_id]
    if missing_clip_ids:
        raise ValueError(
            "metadata does not contain usable rows for selected clip_id values: "
            f"{missing_clip_ids[:10]} total_missing={len(missing_clip_ids)}"
        )

    temp_path = prepare_output_dir(output_path, overwrite=overwrite)
    try:
        manifest_path = temp_path / manifest_name
        output_rows: list[dict[str, object]] = []
        missing_files: list[str] = []
        with manifest_path.open("w", encoding="utf-8") as manifest:
            for clip_id in clip_ids:
                row = rows_by_clip_id[clip_id]
                video_path = resolve_video_path(row)
                video_path_text = str(video_path)
                if any(char.isspace() for char in video_path_text):
                    raise ValueError(
                        "V-JEPA2 space-delimited manifest cannot represent paths "
                        f"with whitespace: {video_path_text}"
                    )
                if check_files and not video_path.is_file():
                    missing_files.append(video_path_text)
                    if len(missing_files) >= 10:
                        break
                manifest.write(f"{video_path_text} {int(label)}\n")

                output_row = dict(row)
                output_row.update(
                    {
                        "vjepa2_ok": 1,
                        "vjepa2_video_path": video_path_text,
                        "vjepa2_manifest_path": str(output_path / manifest_name),
                    }
                )
                output_rows.append(output_row)

        if missing_files:
            raise FileNotFoundError(f"selected clips are missing video files: {missing_files}")

        parquet_summary = write_parquet_shards(
            output_rows,
            temp_path,
            unit="clip",
            parquet_size=parquet_size,
            write_summary=False,
        )
        (temp_path / "clip_ids.txt").write_text(
            "".join(f"{clip_id}\n" for clip_id in clip_ids),
            encoding="utf-8",
        )
        summary = {
            "created_at": utc_now_iso(),
            "automodel_dir": str(automodel_path) if automodel_path is not None else None,
            "clip_ids_path": str(resolved_clip_ids_path),
            "metadata_dir": str(resolved_metadata_dir),
            "output_dir": str(output_path),
            "manifest_name": manifest_name,
            "manifest_path": str(output_path / manifest_name),
            "label": int(label),
            "row_count": len(output_rows),
            "unique_video_id_count": len({str(row.get("video_id", "")) for row in output_rows}),
            "parquet_size": parquet_size,
            "check_files": check_files,
            "parquet": {
                "shard_count": parquet_summary["shard_count"],
                "total_rows": parquet_summary["total_rows"],
                "shards": parquet_summary["shards"],
            },
        }
        (temp_path / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        commit_output_dir(temp_path, output_path, overwrite=overwrite)
    except Exception:
        shutil.rmtree(temp_path, ignore_errors=True)
        raise

    return summary


def read_clip_ids(path: Path) -> list[str]:
    clip_ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not clip_ids:
        raise ValueError(f"clip_ids file is empty: {path}")
    seen: set[str] = set()
    duplicated: set[str] = set()
    for clip_id in clip_ids:
        if clip_id in seen:
            duplicated.add(clip_id)
        seen.add(clip_id)
    if duplicated:
        raise ValueError(f"clip_ids file contains duplicate ids: {sorted(duplicated)[:10]}")
    return clip_ids


def load_rows_for_clip_ids(metadata_dir: Path, clip_ids: set[str]) -> dict[str, dict[str, object]]:
    rows_by_clip_id: dict[str, dict[str, object]] = {}
    for row in iter_parquet(metadata_dir, unit="clip"):
        clip_id = str(row.get("clip_id", "")).strip()
        if clip_id not in clip_ids or not is_usable_row(row):
            continue
        existing = rows_by_clip_id.get(clip_id)
        if existing is None:
            rows_by_clip_id[clip_id] = dict(row)
            continue
        if str(existing.get("clip_path")) != str(row.get("clip_path")):
            raise ValueError(f"multiple usable rows for clip_id={clip_id} have different clip_path values")
    return rows_by_clip_id


def is_usable_row(row: dict[str, object]) -> bool:
    if not str(row.get("clip_path", "")).strip():
        return False
    if "automodel_ok" not in row:
        return True
    try:
        return int(row["automodel_ok"]) == 1
    except (TypeError, ValueError):
        return False


def resolve_video_path(row: dict[str, object]) -> Path:
    clip_path = Path(str(row["clip_path"]))
    if clip_path.is_absolute():
        return clip_path.expanduser().resolve()
    return join_data_dir(clip_path)


def prepare_output_dir(output_path: Path, *, overwrite: bool) -> Path:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output_dir already exists: {output_path}")
    temp_path = output_path.parent / f".{output_path.name}.tmp-{os.getpid()}"
    shutil.rmtree(temp_path, ignore_errors=True)
    temp_path.mkdir(parents=True)
    return temp_path


def commit_output_dir(temp_path: Path, output_path: Path, *, overwrite: bool) -> None:
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"output_dir already exists: {output_path}")
        shutil.rmtree(output_path)
    temp_path.rename(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a V-JEPA2 train.csv from an AutoModel split dataset, preserving "
            "the exact clip_id list used by AutoModel."
        )
    )
    parser.add_argument(
        "--automodel-dir",
        type=Path,
        default=None,
        help="AutoModel split dataset directory containing clip_ids.txt and clip parquet shards.",
    )
    parser.add_argument(
        "--clip-ids",
        type=Path,
        default=None,
        help="Optional clip_ids.txt path. Requires --metadata-dir when --automodel-dir is not set.",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=None,
        help="Optional parquet metadata directory used to resolve clip_id -> clip_path.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-name", default="train.csv")
    parser.add_argument("--label", type=int, default=0)
    parser.add_argument("--parquet-size", type=int, default=500_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check-files", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = create_vjepa2_manifest_from_automodel_split(
        automodel_dir=args.automodel_dir,
        output_dir=args.output_dir,
        clip_ids_path=args.clip_ids,
        metadata_dir=args.metadata_dir,
        manifest_name=args.manifest_name,
        label=args.label,
        parquet_size=args.parquet_size,
        overwrite=args.overwrite,
        check_files=args.check_files,
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
