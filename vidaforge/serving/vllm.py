from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
from typing import Any
from urllib import error, request

from vidaforge.common.ray import (
    alive_node_resources,
    ray_runtime_env,
    resolve_replicas,
    validate_replicas,
)


_VLLM_HEALTH_TIMEOUT_SEC = 6000


@dataclass(frozen=True, slots=True)
class VLLMServerConfig:
    """Runtime config for one vLLM server process."""

    model_path: str
    model_name: str
    replica_id: int
    port: int
    tp_size: int = 1
    host: str = "0.0.0.0"
    vllm_bin: str = "vllm"
    api_key: str = "EMPTY"
    health_poll_interval_sec: float = 2.0
    extra_args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    log_dir: str | None = None
    allowed_local_media_path: str | None = None


@dataclass(frozen=True, slots=True)
class VLLMServerPoolConfig:
    """Ray-managed vLLM server pool config.

    Each server gets one placement group bundle with ``tp_size`` GPUs. The
    driver process owns non-detached actors, so exiting the driver cleans up the
    vLLM subprocesses instead of leaving long-lived services in the Ray cluster.
    """

    model_path: str
    model_name: str
    replicas: int | str = "auto"
    tp_size: int = 1
    cpu_per_replica: int = 8
    base_port: int = 8100
    host: str = "0.0.0.0"
    vllm_bin: str = "vllm"
    api_key: str = "EMPTY"
    ray_address: str = "auto"
    placement_strategy: str = "STRICT_PACK"
    health_poll_interval_sec: float = 2.0
    extra_args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    log_dir: str | None = None
    allowed_local_media_path: str | None = None


@dataclass(frozen=True, slots=True)
class ServerEndpoint:
    replica_id: int
    base_url: str
    node_ip: str
    port: int
    model_name: str
    log_path: str


def _non_empty(value: str, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _validate_pool_config(config: VLLMServerPoolConfig) -> None:
    _non_empty(config.model_path, name="model_path")
    _non_empty(config.model_name, name="model_name")
    validate_replicas(config.replicas)
    if config.tp_size <= 0:
        raise ValueError("tp_size must be > 0")
    if config.cpu_per_replica <= 0:
        raise ValueError("cpu_per_replica must be > 0")
    if config.base_port <= 0:
        raise ValueError("base_port must be > 0")
    if config.health_poll_interval_sec <= 0:
        raise ValueError("health_poll_interval_sec must be > 0")


def build_vllm_serve_cmd(config: VLLMServerConfig) -> list[str]:
    """Build the ``vllm serve`` command for one server process."""
    cmd = [
        config.vllm_bin,
        "serve",
        config.model_path,
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--served-model-name",
        config.model_name,
        "--tensor-parallel-size",
        str(config.tp_size),
    ]
    if config.api_key and config.api_key != "EMPTY":
        cmd.extend(["--api-key", config.api_key])
    if config.allowed_local_media_path:
        cmd.extend(["--allowed-local-media-path", config.allowed_local_media_path])
    cmd.extend(config.extra_args)
    return cmd


def _read_tail(path: str | Path, *, max_lines: int = 80) -> str:
    log_path = Path(path)
    if not log_path.exists():
        return ""
    lines = log_path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def _health_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/v1/models"


def _wait_for_ready(
    *,
    process: subprocess.Popen[bytes],
    port: int,
    api_key: str,
    timeout_sec: int,
    poll_interval_sec: float,
    log_path: str,
) -> None:
    deadline = time.monotonic() + timeout_sec
    url = _health_url(port)
    headers = {}
    if api_key and api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"

    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            tail = _read_tail(log_path)
            raise RuntimeError(
                f"vllm serve exited before ready (returncode={return_code}). "
                f"log_tail:\n{tail}"
            )
        try:
            req = request.Request(url, headers=headers)
            with request.urlopen(req, timeout=5) as response:
                if 200 <= int(response.status) < 500:
                    return
        except (error.URLError, TimeoutError, OSError):
            pass
        time.sleep(poll_interval_sec)

    tail = _read_tail(log_path)
    raise TimeoutError(f"vllm serve did not become ready at {url}. log_tail:\n{tail}")


class VLLMServerActor:
    """Ray actor that owns one ``vllm serve`` subprocess."""

    def __init__(self, config: dict[str, Any]):
        self.config = VLLMServerConfig(**config)
        self.process: subprocess.Popen[bytes] | None = None
        self.log_file: Any = None
        self.log_path = self._build_log_path()

    def _build_log_path(self) -> str:
        root = (
            Path(self.config.log_dir)
            if self.config.log_dir
            else Path(tempfile.gettempdir()) / "vidaforge_vllm"
        )
        root.mkdir(parents=True, exist_ok=True)
        return str(root / f"vllm-replica-{self.config.replica_id:05d}.log")

    def start(self) -> dict[str, object]:
        if self.process is not None and self.process.poll() is None:
            return asdict(self.endpoint())

        import ray

        node_ip = ray.util.get_node_ip_address()
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in self.config.env.items()})
        cmd = build_vllm_serve_cmd(self.config)
        self.log_file = open(self.log_path, "ab", buffering=0)
        self.process = subprocess.Popen(
            cmd,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        _wait_for_ready(
            process=self.process,
            port=self.config.port,
            api_key=self.config.api_key,
            timeout_sec=_VLLM_HEALTH_TIMEOUT_SEC,
            poll_interval_sec=self.config.health_poll_interval_sec,
            log_path=self.log_path,
        )
        return asdict(
            ServerEndpoint(
                replica_id=self.config.replica_id,
                base_url=f"http://{node_ip}:{self.config.port}/v1",
                node_ip=node_ip,
                port=self.config.port,
                model_name=self.config.model_name,
                log_path=self.log_path,
            )
        )

    def endpoint(self) -> ServerEndpoint:
        import ray

        node_ip = ray.util.get_node_ip_address()
        return ServerEndpoint(
            replica_id=self.config.replica_id,
            base_url=f"http://{node_ip}:{self.config.port}/v1",
            node_ip=node_ip,
            port=self.config.port,
            model_name=self.config.model_name,
            log_path=self.log_path,
        )

    def status(self) -> dict[str, object]:
        return {
            "replica_id": self.config.replica_id,
            "pid": None if self.process is None else self.process.pid,
            "returncode": None if self.process is None else self.process.poll(),
            "log_path": self.log_path,
        }

    def log_tail(self, max_lines: int = 80) -> str:
        return _read_tail(self.log_path, max_lines=max_lines)

    def stop(self, timeout_sec: int = 30) -> None:
        process = self.process
        self.process = None
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=timeout_sec)
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None

    def __del__(self) -> None:
        try:
            self.stop(timeout_sec=5)
        except Exception:
            pass


class VLLMServerPool:
    """Context manager that starts Ray-managed vLLM servers and cleans them up."""

    def __init__(self, config: VLLMServerPoolConfig):
        _validate_pool_config(config)
        self.config = config
        self.endpoints: list[ServerEndpoint] = []
        self._actors: list[object] = []
        self._placement_groups: list[object] = []
        self._ray_was_initialized = False
        self._replicas: int | None = None

    @property
    def base_urls(self) -> tuple[str, ...]:
        return tuple(endpoint.base_url for endpoint in self.endpoints)

    @property
    def model(self) -> str:
        return self.config.model_name

    def __enter__(self) -> "VLLMServerPool":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()

    def _server_config(self, replica_id: int) -> VLLMServerConfig:
        return VLLMServerConfig(
            model_path=self.config.model_path,
            model_name=self.config.model_name,
            replica_id=replica_id,
            port=self.config.base_port + replica_id,
            tp_size=self.config.tp_size,
            host=self.config.host,
            vllm_bin=self.config.vllm_bin,
            api_key=self.config.api_key,
            health_poll_interval_sec=self.config.health_poll_interval_sec,
            extra_args=self.config.extra_args,
            env=self.config.env,
            log_dir=self.config.log_dir,
            allowed_local_media_path=self.config.allowed_local_media_path,
        )

    def start(self) -> list[ServerEndpoint]:
        import ray
        from ray.util.placement_group import placement_group
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        if self.endpoints:
            return self.endpoints

        self._ray_was_initialized = ray.is_initialized()
        if not self._ray_was_initialized:
            ray.init(address=self.config.ray_address, ignore_reinit_error=True)

        replicas = resolve_replicas(
            self.config.replicas,
            cluster_resources=ray.cluster_resources(),
            node_resources=alive_node_resources(ray.nodes()),
            cpus_per_replica=self.config.cpu_per_replica,
            gpus_per_replica=self.config.tp_size,
        )
        self._replicas = replicas
        RemoteActor = ray.remote(runtime_env=ray_runtime_env())(VLLMServerActor)
        try:
            for replica_id in range(replicas):
                pg = placement_group(
                    [
                        {
                            "CPU": self.config.cpu_per_replica,
                            "GPU": self.config.tp_size,
                        }
                    ],
                    strategy=self.config.placement_strategy,
                )
                ray.get(pg.ready())
                self._placement_groups.append(pg)

                scheduling_strategy = PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=0,
                    placement_group_capture_child_tasks=True,
                )
                actor = RemoteActor.options(
                    num_cpus=self.config.cpu_per_replica,
                    num_gpus=self.config.tp_size,
                    scheduling_strategy=scheduling_strategy,
                ).remote(asdict(self._server_config(replica_id)))
                self._actors.append(actor)

            endpoint_dicts = ray.get([actor.start.remote() for actor in self._actors])
            self.endpoints = [
                ServerEndpoint(**endpoint_dict) for endpoint_dict in endpoint_dicts
            ]
            return self.endpoints
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        if not self._actors and not self._placement_groups:
            return

        import ray
        from ray.util.placement_group import remove_placement_group

        actors = list(self._actors)
        placement_groups = list(self._placement_groups)
        self._actors.clear()
        self._placement_groups.clear()
        self.endpoints = []

        stop_refs = []
        for actor in actors:
            try:
                stop_refs.append(actor.stop.remote())
            except Exception:
                pass
        if stop_refs:
            try:
                ray.get(stop_refs, timeout=60)
            except Exception:
                pass

        for actor in actors:
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                pass

        for pg in placement_groups:
            try:
                remove_placement_group(pg)
            except Exception:
                pass

        if not self._ray_was_initialized and ray.is_initialized():
            ray.shutdown()
