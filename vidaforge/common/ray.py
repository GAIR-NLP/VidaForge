from __future__ import annotations

from collections.abc import Iterable
import os
from pathlib import Path


_RAY_REQUIRED_PATH_ENV_VARS = ("DATA_DIR",)
_RAY_OPTIONAL_PATH_ENV_VARS = ("RAW_DIR",)
_RAY_OPTIONAL_RAW_ENV_VARS = ("LD_LIBRARY_PATH",)


def resolve_max_pending_tasks(
    *,
    total_cpus: int | float,
    cpus_per_task: int | float,
    pending_multiplier: int = 4
) -> int:
    if total_cpus <= 0 or cpus_per_task <= 0:
        return 1
    return max(1, int(total_cpus // cpus_per_task) * pending_multiplier) 


def validate_replicas(replicas: int | str) -> None:
    if isinstance(replicas, str):
        if replicas.strip().lower() == "auto":
            return
        raise ValueError("replicas must be a positive integer or 'auto'")
    if int(replicas) <= 0:
        raise ValueError("replicas must be > 0")


def alive_node_resources(nodes: Iterable[dict[str, object]]) -> list[dict[str, float]]:
    resources: list[dict[str, float]] = []
    for node in nodes:
        if not bool(node.get("Alive", False)):
            continue
        raw_resources = node.get("Resources", {})
        if not isinstance(raw_resources, dict):
            continue
        resources.append(
            {
                str(name): float(quantity)
                for name, quantity in raw_resources.items()
            }
        )
    return resources


def _replica_slots_for_resources(
    resources: dict[str, float],
    *,
    cpus_per_replica: int | float,
    gpus_per_replica: int | float,
) -> int:
    by_cpu = int(float(resources.get("CPU", 0.0)) // float(cpus_per_replica))
    if gpus_per_replica > 0:
        by_gpu = int(float(resources.get("GPU", 0.0)) // float(gpus_per_replica))
        return min(by_cpu, by_gpu)
    return by_cpu


def _resource_limit_for_replicas(
    *,
    cluster_resources: dict[str, float],
    node_resources: list[dict[str, float]] | None,
    cpus_per_replica: int | float,
    gpus_per_replica: int | float,
) -> int:
    if node_resources:
        return sum(
            _replica_slots_for_resources(
                resources,
                cpus_per_replica=cpus_per_replica,
                gpus_per_replica=gpus_per_replica,
            )
            for resources in node_resources
        )
    return _replica_slots_for_resources(
        cluster_resources,
        cpus_per_replica=cpus_per_replica,
        gpus_per_replica=gpus_per_replica,
    )


def resolve_replicas(
    replicas: int | str,
    *,
    cluster_resources: dict[str, float],
    node_resources: list[dict[str, float]] | None = None,
    cpus_per_replica: int | float,
    gpus_per_replica: int | float,
    max_replicas: int | None = None,
) -> int:
    validate_replicas(replicas)
    if cpus_per_replica <= 0:
        raise ValueError("cpus_per_replica must be > 0")
    if gpus_per_replica < 0:
        raise ValueError("gpus_per_replica must be >= 0")

    resource_limit = _resource_limit_for_replicas(
        cluster_resources=cluster_resources,
        node_resources=node_resources,
        cpus_per_replica=cpus_per_replica,
        gpus_per_replica=gpus_per_replica,
    )

    if resource_limit <= 0:
        raise RuntimeError(
            "insufficient Ray resources for replicas "
            f"(cluster_resources={cluster_resources}, "
            f"node_resources={node_resources}, "
            f"cpus_per_replica={cpus_per_replica}, "
            f"gpus_per_replica={gpus_per_replica})"
        )

    work_limit = int(max_replicas) if max_replicas is not None else None
    is_auto = isinstance(replicas, str) and replicas.strip().lower() == "auto"
    if is_auto:
        resolved = resource_limit
        if work_limit is not None:
            resolved = min(resolved, work_limit)
        if resolved <= 0:
            raise RuntimeError(
                "insufficient work units for replicas=auto "
                f"(max_replicas={max_replicas})"
            )
        return resolved

    requested = int(replicas)
    if requested > resource_limit:
        raise RuntimeError(
            f"requested replicas={requested} exceeds available Ray resources "
            f"(max_replicas_by_resources={resource_limit}, "
            f"cluster_resources={cluster_resources}, "
            f"node_resources={node_resources}, "
            f"cpus_per_replica={cpus_per_replica}, "
            f"gpus_per_replica={gpus_per_replica})"
        )
    if work_limit is not None and requested > work_limit:
        raise RuntimeError(
            f"requested replicas={requested} exceeds available work units "
            f"(max_replicas_by_work={work_limit})"
        )
    return requested


def runtime_env_vars() -> dict[str, str]:
    env_vars: dict[str, str] = {}
    for name in _RAY_REQUIRED_PATH_ENV_VARS:
        try:
            value = os.environ[name]
        except KeyError as exc:
            raise RuntimeError(f"{name} must be set for Ray workers.") from exc
        env_vars[name] = str(Path(value).expanduser().resolve())
    for name in _RAY_OPTIONAL_PATH_ENV_VARS:
        if value := os.environ.get(name):
            env_vars[name] = str(Path(value).expanduser().resolve())
    for name in _RAY_OPTIONAL_RAW_ENV_VARS:
        if value := os.environ.get(name):
            env_vars[name] = value
    return env_vars


def ray_runtime_env() -> dict[str, dict[str, str]]:
    return {"env_vars": runtime_env_vars()}
