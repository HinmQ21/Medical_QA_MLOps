"""HTTP client the API uses to call the retrieval service /search endpoint."""

import os

import httpx


class RetrievalClient:
    def __init__(
        self,
        base_url: str,
        client: httpx.Client | None = None,
        timeout: float = 10.0,
    ):
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout)

    @classmethod
    def from_env(cls) -> "RetrievalClient":
        return cls(base_url=os.environ.get("RETRIEVAL_URL", "http://localhost:8001"))

    def search(self, query: str, top_k: int) -> list[str]:
        resp = self._client.post(
            f"{self.base_url}/search",
            json={"queries": [query], "top_k": top_k},
        )
        resp.raise_for_status()
        payload = resp.json()
        if not payload:
            return []
        return payload[0]["results"]
