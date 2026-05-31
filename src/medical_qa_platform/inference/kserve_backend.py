"""KServe CPU smoke/mock backend (v1 predict protocol)."""

import os

import httpx

from .base import ModelBackend


class KServeBackend(ModelBackend):
    name = "kserve"

    def __init__(
        self,
        url: str,
        client: httpx.Client | None = None,
        timeout: float = 30.0,
    ):
        self.url = url
        self._client = client or httpx.Client(timeout=timeout)

    @classmethod
    def from_env(cls) -> "KServeBackend":
        return cls(url=os.environ.get("KSERVE_URL", ""))

    def generate(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> str:
        resp = self._client.post(
            self.url,
            json={"instances": [{"messages": messages}]},
        )
        resp.raise_for_status()
        return resp.json()["predictions"][0]["text"]
