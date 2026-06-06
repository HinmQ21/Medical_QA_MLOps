"""Model backend abstraction."""

import os

from .base import ModelBackend


def get_backend(name: str | None = None) -> ModelBackend:
    """Construct a backend by name (defaults to env MODEL_BACKEND, then mock)."""
    name = (name or os.environ.get("MODEL_BACKEND", "mock")).lower()
    if name == "mock":
        from .mock_backend import MockBackend

        return MockBackend()
    if name == "vllm":
        from .vllm_backend import VllmBackend

        return VllmBackend.from_env()
    if name == "kserve":
        from .kserve_backend import KServeBackend

        return KServeBackend.from_env()
    raise ValueError(f"unknown MODEL_BACKEND: {name!r}")
