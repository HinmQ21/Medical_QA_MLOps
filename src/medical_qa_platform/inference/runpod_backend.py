"""RunPod vLLM backend via the OpenAI-compatible chat-completions API."""

import os

import httpx

from .base import ModelBackend


class RunpodBackend(ModelBackend):
    name = "runpod"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        client: httpx.Client | None = None,
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self._client = client or httpx.Client(timeout=timeout)

    @classmethod
    def from_env(cls) -> "RunpodBackend":
        return cls(
            base_url=os.environ.get("RUNPOD_BASE_URL", ""),
            model=os.environ.get("RUNPOD_MODEL", ""),
            api_key=os.environ.get("RUNPOD_API_KEY", ""),
        )

    def generate(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> str:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = self._client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
