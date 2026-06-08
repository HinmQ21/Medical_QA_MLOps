"""Pure HTTP client for the Medical QA /predict API.

No Streamlit import — importable and unit-testable in the base venv. The UI
(``app.streamlit_app``) is the only Streamlit-aware module.
"""

from __future__ import annotations

from dataclasses import dataclass
import json

import httpx


class PredictError(Exception):
    """A human-readable failure talking to the API (surfaced in the UI)."""


@dataclass
class PredictResult:
    answer: str | None
    raw_output: str
    evidence: list[str]
    trace: list[dict]
    backend: str
    model_version: str
    contract_version: str
    latency_ms: float
    trace_id: str


def build_payload(question: str) -> dict:
    """Validate the UI input into the /predict request body (mirrors the server contract)."""
    question = question.strip()
    if not question:
        raise ValueError("Câu hỏi không được để trống.")
    return {"question": question}


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
    conn = client or httpx.Client()
    try:
        resp = conn.request(method, url, headers=_headers(api_key), timeout=timeout, **kwargs)
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
    timeout: float = 120.0,
    client: httpx.Client | None = None,
) -> PredictResult:
    resp = _request("POST", base_url, "/predict", api_key, timeout, client, json=payload)
    try:
        data = resp.json()
    except ValueError as exc:  # json.JSONDecodeError is a subclass of ValueError
        raise PredictError("Phản hồi không hợp lệ — server returned non-JSON.") from exc
    return PredictResult(
        answer=data.get("answer"),
        raw_output=data.get("raw_output", ""),
        evidence=data.get("evidence", []),
        trace=data.get("trace", []),
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
    try:
        return resp.json()
    except ValueError as exc:  # json.JSONDecodeError is a subclass of ValueError
        raise PredictError("Phản hồi không hợp lệ — server returned non-JSON.") from exc


def build_transcript_blocks(trace: list[dict]) -> list[tuple[str, str]]:
    """Turn the API trace into (label, body) blocks for display.

    Assistant turns are labelled with the search query they issued (if any);
    tool turns are labelled as KG results. Pure — no Streamlit import — so it is
    unit-testable in the base venv.
    """
    blocks: list[tuple[str, str]] = []
    for turn in trace:
        role = turn.get("role")
        content = turn.get("content") or ""
        if role == "tool":
            blocks.append(("📚 Kết quả tri thức (KG)", content))
            continue
        label = "🤖 Trợ lý"
        calls = turn.get("tool_calls") or []
        if calls:
            queries = []
            for call in calls:
                args = call.get("function", {}).get("arguments", "")
                try:
                    parsed = json.loads(args) if isinstance(args, str) else args
                    query = parsed.get("query", "") if isinstance(parsed, dict) else ""
                except (json.JSONDecodeError, AttributeError, TypeError):
                    query = args if isinstance(args, str) else ""
                queries.append(str(query))
            label = "🤖 Trợ lý · 🔎 search_medical_knowledge(" + "; ".join(queries) + ")"
        blocks.append((label, content))
    return blocks
