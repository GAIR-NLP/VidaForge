from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
from typing import Iterator


DEFAULT_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
_DATA_DIR_ENV = "DATA_DIR"
_RAW_DIR_ENV = "RAW_DIR"
_PIPELINE_DATA_DIRS = {
    "stage1_ingestion",
    "stage2_segmentation",
    "stage3_selection",
    "stage4_annotation",
    "stage5_packaging",
}


def replace_path_part(
    path: str | Path,
    *,
    old: str,
    new: str,
) -> Path:
    """Replace one path component by exact component name."""
    source_path = Path(path)
    if not old:
        raise ValueError("old path component is required")
    if not new:
        raise ValueError("new path component is required")

    parts = list(source_path.parts)
    try:
        part_index = parts.index(old)
    except ValueError as exc:
        raise ValueError(
            f"Cannot replace path component {old!r} in path={source_path}: "
            "component not found."
        ) from exc
    parts[part_index] = new
    return Path(*parts)


def _env_dir(env_name: str) -> Path:
    try:
        value = os.environ[env_name]
    except KeyError as exc:
        raise RuntimeError(f"{env_name} must be set to resolve pipeline paths.") from exc
    return Path(value).expanduser().resolve()


def _strip_env_dir(path: str | Path, *, env_name: str) -> str:
    return str(Path(path).expanduser().resolve().relative_to(_env_dir(env_name)))


def _join_env_dir(path: str | Path, *, env_name: str) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        raise ValueError(f"path must be relative to {env_name}: {path_obj}")
    return _env_dir(env_name) / path_obj


def strip_data_dir(path: str | Path) -> str:
    return _strip_env_dir(path, env_name=_DATA_DIR_ENV)


def join_data_dir(path: str | Path) -> Path:
    return _join_env_dir(path, env_name=_DATA_DIR_ENV)


def strip_raw_dir(path: str | Path) -> str:
    return _strip_env_dir(path, env_name=_RAW_DIR_ENV)


def join_raw_dir(path: str | Path) -> Path:
    return _join_env_dir(path, env_name=_RAW_DIR_ENV)


def _is_safe_tar_member_path(member_path: str) -> bool:
    if not member_path:
        return False
    path = Path(member_path)
    if path.is_absolute():
        return False
    return ".." not in path.parts


def scan_tar(
    tar_path: str | Path,
    *,
    video_extensions: set[str] | None = None,
) -> Iterator[str]:
    """Yield video member paths from an uncompressed tar archive."""
    extensions = {
        ext.lower() for ext in (video_extensions or DEFAULT_VIDEO_EXTENSIONS)
    }
    with tarfile.open(Path(tar_path).expanduser().resolve(), "r:") as archive:
        for member in archive:
            if not member.isfile():
                continue
            if not _is_safe_tar_member_path(member.name):
                continue
            if Path(member.name).suffix.lower() not in extensions:
                continue
            yield member.name


def _skip_raw_video_scan_dir(path: Path, *, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return False
    if not parts:
        return False
    if len(parts) == 1 and parts[0] == "meta":
        return True
    if len(parts) == 2 and parts[0] == "data" and parts[1] in _PIPELINE_DATA_DIRS:
        return True
    return False


def scan_raw_videos(
    root_dir: str | Path,
    *,
    video_extensions: set[str] | None = None,
) -> Iterator[dict[str, object]]:
    """Scan ordinary video files and video members inside .tar archives."""
    root = Path(root_dir).expanduser().resolve()
    extensions = {
        ext.lower() for ext in (video_extensions or DEFAULT_VIDEO_EXTENSIONS)
    }

    for dir_path, dir_names, file_names in os.walk(root):
        current_dir = Path(dir_path)
        dir_names[:] = sorted(
            name
            for name in dir_names
            if not _skip_raw_video_scan_dir(current_dir / name, root=root)
        )
        for file_name in sorted(file_names):
            path = current_dir / file_name
            suffix = path.suffix.lower()
            if suffix in extensions:
                yield {
                    "raw_type": "file",
                    "raw_path": path,
                    "raw_member_path": "",
                }
                continue
            if suffix == ".tar":
                for member_path in scan_tar(path, video_extensions=extensions):
                    yield {
                        "raw_type": "tar",
                        "raw_path": path,
                        "raw_member_path": member_path,
                    }


def _resolve_temp_dir(temp_dir: str | Path | None) -> Path:
    root = (
        Path(temp_dir)
        if temp_dir is not None
        else Path(tempfile.gettempdir()) / "raw_video"
    )
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


@contextmanager
def get_raw_video_path(
    row: dict[str, object],
    *,
    temp_dir: str | Path | None = None,
) -> Iterator[Path]:
    """Yield a local file path that ffprobe/ffmpeg can read for a raw video row."""
    raw_type = str(row.get("raw_type", "file"))
    raw_path = join_raw_dir(str(row["raw_path"]))

    if raw_type == "file":
        yield raw_path
        return

    if raw_type == "tar":
        member_path = str(row.get("raw_member_path", ""))
        if not _is_safe_tar_member_path(member_path):
            raise ValueError(f"invalid tar member path: {member_path!r}")

        temp_path: Path | None = None
        try:
            with tarfile.open(raw_path, "r:") as archive:
                try:
                    member = archive.getmember(member_path)
                except KeyError as exc:
                    raise ValueError(f"tar member not found: {member_path}") from exc
                if not member.isfile():
                    raise ValueError(f"tar member is not a file: {member_path}")

                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"cannot extract tar member: {member_path}")

                with tempfile.NamedTemporaryFile(
                    dir=_resolve_temp_dir(temp_dir),
                    suffix=Path(member_path).suffix,
                    delete=False,
                ) as temp_file:
                    temp_path = Path(temp_file.name)
                    shutil.copyfileobj(extracted, temp_file)
        except Exception:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            raise

        try:
            yield temp_path
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        return

    raise ValueError(f"unsupported raw_type: {raw_type}")
