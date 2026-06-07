"""Pure HTTP client for the Medical QA /predict API.

No Streamlit import — importable and unit-testable in the base venv. The UI
(``app.streamlit_app``) is the only Streamlit-aware module.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx


class PredictError(Exception):
    """A human-readable failure talking to the API (surfaced in the UI)."""


@dataclass
class PredictResult:
    answer: str | None
    evidence: list[str]
    backend: str
    model_version: str
    contract_version: str
    latency_ms: float
    trace_id: str


def build_payload(question: str, options: dict[str, str]) -> dict:
    """Validate UI inputs into the /predict request body (mirrors the server contract)."""
    question = question.strip()
    if not question:
        raise ValueError("Câu hỏi không được để trống.")
    clean = {key: value.strip() for key, value in options.items() if value.strip()}
    if not (2 <= len(clean) <= 10):
        raise ValueError("Cần 2–10 phương án không rỗng.")
    for key in clean:
        if len(key) != 1 or not ("A" <= key <= "Z"):
            raise ValueError(f"Khóa phương án {key!r} phải là một chữ cái A–Z.")
    return {"question": question, "options": clean}


def _headers(api_key: str | None) -> dict[str, str]:
    return {"x-api-key": api_key} if api_key else {}


def _request(
    method: str,
    base_url: str,
    path: str,
    api_key: str | None,
    timeout: float,
    client: httpx.Client | None,
    **kwargs,
) -> httpx.Response:
    url = base_url.rstrip("/") + path
    owned = client is None
    conn = client or httpx.Client(timeout=timeout)
    try:
        resp = conn.request(method, url, headers=_headers(api_key), **kwargs)
    except httpx.TimeoutException as exc:
        raise PredictError(f"Hết thời gian chờ — timed out after {timeout:g}s.") from exc
    except httpx.HTTPError as exc:
        raise PredictError("Không thể kết nối — could not reach the API.") from exc
    finally:
        if owned:
            conn.close()
    if resp.status_code == 401:
        raise PredictError("Bị từ chối — unauthorized (401), kiểm tra API key của gateway.")
    if resp.status_code != 200:
        raise PredictError(f"API trả về HTTP {resp.status_code}.")
    return resp


def predict(
    base_url: str,
    api_key: str | None,
    payload: dict,
    timeout: float = 30.0,
    client: httpx.Client | None = None,
) -> PredictResult:
    resp = _request("POST", base_url, "/predict", api_key, timeout, client, json=payload)
    data = resp.json()
    return PredictResult(
        answer=data.get("answer"),
        evidence=data.get("evidence", []),
        backend=data.get("backend", ""),
        model_version=data.get("model_version", ""),
        contract_version=data.get("contract_version", ""),
        latency_ms=data.get("latency_ms", 0.0),
        trace_id=data.get("trace_id", ""),
    )


def fetch_version(
    base_url: str,
    api_key: str | None,
    timeout: float = 5.0,
    client: httpx.Client | None = None,
) -> dict:
    resp = _request("GET", base_url, "/version", api_key, timeout, client)
    return resp.json()
