from __future__ import annotations

from omegaconf import DictConfig, ListConfig, OmegaConf


def optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "null":
        return None
    return float(value)


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def plain_dict(value: object) -> dict[str, object]:
    data = OmegaConf.to_container(value, resolve=True)
    if not isinstance(data, dict):
        raise TypeError(f"expected dict config, got {type(data)!r}")
    return dict(data)


def string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    value = (
        OmegaConf.to_container(value, resolve=True)
        if isinstance(value, (DictConfig, ListConfig))
        else value
    )
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return (str(value),)


def string_dict(value: object) -> dict[str, str]:
    if value is None:
        return {}
    value = (
        OmegaConf.to_container(value, resolve=True)
        if isinstance(value, (DictConfig, ListConfig))
        else value
    )
    return {str(key): str(item) for key, item in dict(value).items()}


def object_dict(value: object) -> dict[str, object]:
    if value is None:
        return {}
    value = (
        OmegaConf.to_container(value, resolve=True)
        if isinstance(value, (DictConfig, ListConfig))
        else value
    )
    return dict(value)


def optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "null":
        return None
    return text
