"""Runtime configuration read from environment variables."""

import os
from dataclasses import dataclass


@dataclass
class Settings:
    model_backend: str = "mock"
    retrieval_url: str = "http://localhost:8001"
    model_version: str = "dev"
    top_k: int = 5
    drift_log_path: str = "drift_log.jsonl"
    max_tokens: int = 512
    max_tool_iterations: int = 2

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model_backend=os.environ.get("MODEL_BACKEND", "mock"),
            retrieval_url=os.environ.get("RETRIEVAL_URL", "http://localhost:8001"),
            model_version=os.environ.get("MODEL_VERSION", "dev"),
            top_k=int(os.environ.get("TOP_K", "5")),
            drift_log_path=os.environ.get("DRIFT_LOG_PATH", "drift_log.jsonl"),
            max_tokens=int(os.environ.get("MAX_TOKENS", "512")),
            max_tool_iterations=int(os.environ.get("MAX_TOOL_ITERATIONS", "2")),
        )
