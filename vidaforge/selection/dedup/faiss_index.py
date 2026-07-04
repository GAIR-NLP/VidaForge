from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


FaissIndexDevice = Literal["cpu", "gpu", "gpu_cuvs"]
FlatIPIndexBackend = Literal["faiss_auto", "faiss_cpu", "faiss_gpu", "gpu_cuvs"]


@dataclass(slots=True)
class FaissIndex:
    index: Any
    device: FaissIndexDevice
    gpu_resources: Any | None = None


class CuvsFlatIPIndex:
    def __init__(self, faiss: Any, vectors: np.ndarray, *, device: int = 0) -> None:
        self.faiss = faiss
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.device = int(device)
        self.resources = faiss.StandardGpuResources()

    def search(self, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        return self.faiss.knn_gpu(
            self.resources,
            queries,
            self.vectors,
            int(top_k),
            metric=self.faiss.METRIC_INNER_PRODUCT,
            device=self.device,
            use_cuvs=True,
        )


def build_flat_ip_index(
    vectors: np.ndarray,
    *,
    use_gpu: bool,
    use_float16: bool = False,
    backend: FlatIPIndexBackend = "faiss_auto",
) -> FaissIndex:
    import faiss

    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    dim = int(vectors.shape[1])
    resolved_backend = _resolve_flat_ip_backend(backend, use_gpu=use_gpu)
    if resolved_backend == "gpu_cuvs":
        return _build_gpu_cuvs_flat_ip_index(faiss, vectors)
    if resolved_backend == "faiss_gpu":
        return _build_gpu_flat_ip_index(
            faiss,
            vectors,
            dim=dim,
            use_float16=use_float16,
        )

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return FaissIndex(index=index, device="cpu")


def _resolve_flat_ip_backend(
    backend: FlatIPIndexBackend,
    *,
    use_gpu: bool,
) -> Literal["faiss_cpu", "faiss_gpu", "gpu_cuvs"]:
    if backend == "faiss_auto":
        return "faiss_gpu" if use_gpu else "faiss_cpu"
    return backend


def build_binary_flat_index(
    hashes: np.ndarray,
    *,
    bits: int,
    use_gpu: bool,
) -> FaissIndex:
    import faiss

    hashes = np.ascontiguousarray(hashes, dtype=np.uint8)
    if use_gpu:
        return _build_gpu_binary_flat_index(faiss, hashes, bits=bits)

    index = faiss.IndexBinaryFlat(bits)
    index.add(hashes)
    return FaissIndex(index=index, device="cpu")


def _build_gpu_flat_ip_index(
    faiss: Any,
    vectors: np.ndarray,
    *,
    dim: int,
    use_float16: bool,
) -> FaissIndex:
    _require_faiss_gpu(
        faiss,
        ["StandardGpuResources", "GpuIndexFlatIP", "GpuIndexFlatConfig"],
    )
    resources = faiss.StandardGpuResources()
    config = faiss.GpuIndexFlatConfig()
    config.device = 0
    if hasattr(config, "useFloat16"):
        config.useFloat16 = bool(use_float16)

    index = faiss.GpuIndexFlatIP(resources, dim, config)
    index.add(vectors)
    return FaissIndex(index=index, device="gpu", gpu_resources=resources)


def _build_gpu_cuvs_flat_ip_index(faiss: Any, vectors: np.ndarray) -> FaissIndex:
    _require_faiss_gpu(
        faiss,
        ["StandardGpuResources", "knn_gpu", "METRIC_INNER_PRODUCT"],
    )
    if not hasattr(faiss, "GpuIndexCagra"):
        raise RuntimeError("FAISS cuVS symbols are missing: ['GpuIndexCagra']")
    index = CuvsFlatIPIndex(faiss, vectors)
    return FaissIndex(index=index, device="gpu_cuvs", gpu_resources=index.resources)


def _build_gpu_binary_flat_index(
    faiss: Any,
    hashes: np.ndarray,
    *,
    bits: int,
) -> FaissIndex:
    _require_faiss_gpu(
        faiss,
        ["StandardGpuResources", "GpuIndexBinaryFlat", "GpuIndexBinaryFlatConfig"],
    )
    resources = faiss.StandardGpuResources()
    config = faiss.GpuIndexBinaryFlatConfig()
    config.device = 0

    index = faiss.GpuIndexBinaryFlat(resources, bits, config)
    index.add(hashes)
    return FaissIndex(index=index, device="gpu", gpu_resources=resources)


def _require_faiss_gpu(faiss: Any, symbols: list[str]) -> None:
    missing = [symbol for symbol in symbols if not hasattr(faiss, symbol)]
    if missing:
        raise RuntimeError(f"FAISS GPU symbols are missing: {missing}")
    if not hasattr(faiss, "get_num_gpus"):
        raise RuntimeError("FAISS get_num_gpus() is missing.")
    if int(faiss.get_num_gpus()) <= 0:
        raise RuntimeError("FAISS GPU index requested, but FAISS reports 0 GPUs.")
