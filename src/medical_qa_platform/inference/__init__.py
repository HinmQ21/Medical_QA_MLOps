"""Model backend abstraction."""

import os

from .base import ModelBackend


def get_backend(name: str | None = None) -> ModelBackend:
    """Construct a backend by name (defaults to env MODEL_BACKEND, then mock)."""
    name = (name or os.environ.get("MODEL_BACKEND", "mock")).lower()
    if name == "mock":
        from .mock_backend import MockBackend

        return MockBackend()
    if name in ("llm", "vllm"):  # "vllm" is a back-compat alias for the generic LLM/OpenAI backend
        from .llm_backend import LLMBackend

        return LLMBackend.from_env()
    raise ValueError(f"unknown MODEL_BACKEND: {name!r}")
