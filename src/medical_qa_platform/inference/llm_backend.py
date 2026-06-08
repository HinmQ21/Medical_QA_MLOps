"""Generic OpenAI-compatible chat-completions backend.

Talks to any server exposing the OpenAI ``/v1`` API — a self-hosted vLLM server
on the DGX-Spark (reached over a Cloudflare Tunnel) or an in-cluster llama.cpp
server, among others. Configured via the ``LLM_BASE_URL``, ``LLM_MODEL`` and
``LLM_API_KEY`` environment variables.
"""

import os

import httpx

from .base import ModelBackend


class LLMBackend(ModelBackend):
    name = "llm"

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
    def from_env(cls) -> "LLMBackend":
        return cls(
            base_url=os.environ.get("LLM_BASE_URL", ""),
            model=os.environ.get("LLM_MODEL", ""),
            api_key=os.environ.get("LLM_API_KEY", ""),
        )

    def _auth_headers(self) -> dict:
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    def generate(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> str:
        resp = self._client.post(
            f"{self.base_url}/chat/completions",
            headers=self._auth_headers(),
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def health_check(self) -> bool:
        try:
            resp = self._client.get(
                f"{self.base_url}/models",
                headers=self._auth_headers(),
                timeout=5.0,
            )
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
